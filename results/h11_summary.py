"""Print held-out | confirm side-by-side summaries from results/h11_*.json (R5-C token efficiency). Usage: python results/h11_summary.py [pipe|doc] [mu_tag]"""
import json, sys, os

R = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def pipe(tag=""):
    h, c = load(f"h11_pipe{tag}.json"), load(f"h11_pipe{tag}_confirm61-90.json")
    print(f"##### world P (g18 copy)  mu tag '{tag}'  tuning picks held: R*={h['tuning']['rv']} theta*={h['tuning']['theta']} vg={h['tuning']['use_vg']} rg={h['tuning']['use_rg']}")
    print("arm            | group A: dq  cost%  dU | group B: dq cost% dU | controls: dq cost% dU   (held | confirm)  [EQ holm|raw|NI /n]")
    arms = [a['arm'] for a in h['groups_vs_L3']]
    for a in arms:
        gh = next(x for x in h['groups_vs_L3'] if x['arm'] == a)['groups']
        gc = next(x for x in c['groups_vs_L3'] if x['arm'] == a)['groups']
        line = f"{a:<17}"
        for g in ('A', 'B', 'ctl'):
            x, y = gh[g], gc[g]
            line += f"| {x['dq']:+.4f} {-x['cost_cut_pct']:+6.1f}% {x['dU03']:+.4f} [{x['eq_q_holm']}|{x['eq_q_raw']}|{x.get('ni_q','?')}/{x['n']}] / {y['dq']:+.4f} {-y['cost_cut_pct']:+6.1f}% {y['dU03']:+.4f} [{y['eq_q_holm']}|{y['eq_q_raw']}|{y.get('ni_q','?')}/{y['n']}] "
        print(line)


def doc(tag=""):
    h, c = load(f"h11_doc{tag}.json"), load(f"h11_doc{tag}_confirm61-90.json")
    print(f"##### world D (doc QA on g16 fixtures)  picks: {h['picks']}")
    arms = [a['arm'] for a in h['groups_vs_D0']]
    for a in arms:
        gh = next(x for x in h['groups_vs_D0'] if x['arm'] == a)['groups']
        gc = next(x for x in c['groups_vs_D0'] if x['arm'] == a)['groups']
        line = f"{a:<20}"
        for g in ('treatment', 'controls'):
            x, y = gh[g], gc[g]
            line += f"| {x['dsucc']:+.4f} {-x['cost_cut_pct']:+6.1f}% {x['dU03']:+.4f} [{x['eq_holm']}|{x['eq_raw']}|{x.get('ni','?')}/{x['n']}] / {y['dsucc']:+.4f} {-y['cost_cut_pct']:+6.1f}% {y['dU03']:+.4f} [{y['eq_holm']}|{y['eq_raw']}|{y.get('ni','?')}/{y['n']}] "
        print(line)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'pipe'
    tag = sys.argv[2] if len(sys.argv) > 2 else ''
    (pipe if which == 'pipe' else doc)(tag)
