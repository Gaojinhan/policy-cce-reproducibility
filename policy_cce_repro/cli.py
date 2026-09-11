"""Offline result reproduction and an explicit, bounded synthetic demo."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('verify', 'audits', 'outcomes', 'paper'):
        command = sub.add_parser(name)
        command.add_argument('--data', type=Path, required=True)
        if name != 'verify':
            command.add_argument('--output', type=Path, required=True)
            command.add_argument('--workers', type=int, default=1)
        if name == 'audits':
            command.add_argument('--job-id', action='append')
    command=sub.add_parser('demo', help='Run a small synthetic example, not a formal experiment')
    command.add_argument('--output',type=Path,required=True)
    command=sub.add_parser('unpack', help='Verify and extract a downloaded release archive locally')
    command.add_argument('--archive',type=Path,required=True)
    command.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    # Set before NumPy/SciPy imports. Statistical parallelism is explicit.
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    from policy_cce_repro.offline import DataError, Dataset
    try:
        if args.command=='demo':
            from policy_cce_repro.demo import run_demo
            report=run_demo(args.output)
        elif args.command=='unpack':
            from policy_cce_repro.unpack import unpack_data
            report=unpack_data(args.archive,args.output)
        elif args.command == 'verify':
            dataset = Dataset(args.data)
            report = dataset.verify_all()
        else:
            dataset = Dataset(args.data)
            if not 1 <= args.workers <= 4:
                raise DataError('Use 1–4 workers for offline statistics')
            out = dataset.validate_output_dir(args.output)
            # Never reuse a directory that might hold earlier scientific output.
            out.mkdir(parents=True, exist_ok=False)
            if args.command == 'audits':
                from policy_cce_repro.recompute_audits import recompute_audits
                report = recompute_audits(dataset, out, workers=args.workers, job_ids=args.job_id)
            elif args.command == 'outcomes':
                from policy_cce_repro.recompute_outcomes import recompute_outcomes
                report = recompute_outcomes(dataset, out, workers=args.workers)
            else:
                from policy_cce_repro.paper import build_paper
                report = build_paper(dataset, out, workers=args.workers)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report.get('status') == 'pass' else 1
    except (DataError, FileExistsError) as error:
        parser.exit(2, 'Offline input/output error: ' + str(error) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
