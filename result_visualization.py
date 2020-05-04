from fastai.vision import plt


def show_results(td, learner):
    size = len(td.test_ds)
    _,axs = plt.subplots(size,3, figsize=(26,35))
    for num, image in enumerate(td.test_ds):
        pred = learner.predict(image[0])[0]
        _ax = axs
        if size > 1:
            _ax = _ax[num]
        image[0].show(ax=_ax[0], title='no mask')
        image[0].show(ax=_ax[1], y=pred, title='masked')
        pred.show(ax=_ax[2], title='mask only', alpha=1.)