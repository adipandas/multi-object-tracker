import os
import motmetrics as mm


def compute_motchallenge(dname):
    df_gt = mm.io.loadtxt(os.path.join(dname, 'gt.txt'))
    df_test = mm.io.loadtxt(os.path.join(dname, 'test.txt'))
    return mm.utils.compare_to_groundtruth(df_gt, df_test, 'iou', distth=0.5)


def metrics_motchallenge_files(data_dir='data'):
    """
    Metric evaluation for sequences TUD-Campus and TUD-Stadtmitte for MOTChallenge.
    """

    dnames = ['TUD-Campus', 'TUD-Stadtmitte']

    # accumulators for two datasets TUD-Campus and TUD-Stadtmitte.
    accs = [compute_motchallenge(os.path.join(data_dir, d)) for d in dnames]

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=dnames, generate_overall=True)

    print()
    print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names, formatters=mh.formatters))


if __name__ == '__main__':
    metrics_motchallenge_files(data_dir='data')
