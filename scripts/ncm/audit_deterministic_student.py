"""Full-shape student-conditioning audit with deterministic execution."""
import argparse
import json
from pathlib import Path
from .audit_canonical_onpolicy import run
from .backbone import digest
from .deterministic_student_policy import enable, validate_repeatability
from .train import write_json


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'selection', 'manifest', 'execution-audit', 'repeatability', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    policy = enable()
    control = json.loads(Path(args.repeatability).read_text())
    validate_repeatability(control, digest(args.checkpoint))
    try:
        run(args)
    finally:
        path = Path(args.output) / 'report.json'
        if path.exists():
            report = json.loads(path.read_text())
            report.update(deterministic_execution=policy, repeatability_control=control,
                          repeatability_report_sha256=digest(args.repeatability))
            report.setdefault('implementation_sha256', {}).update({
                n: digest(Path(__file__).with_name(n)) for n in
                ('audit_deterministic_student.py', 'deterministic_student_policy.py', 'canonical_onpolicy_policy.py')})
            write_json(path, report)
