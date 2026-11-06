import glob
import os
import paddle

root = '/root/autodl-tmp/dior_clean/rtdetr_r50vd_6x_dior_resume21_safe'

files = sorted(
    glob.glob(root + '/*.pdparams'),
    key=os.path.getmtime
)

print('\n========== CHECKPOINT FINITE CHECK ==========')

for p in files:
    state = paddle.load(p)

    bad = []
    bad_bn = []
    bad_non_bn = []

    for name, x in state.items():
        if not isinstance(x, paddle.Tensor):
            continue

        if not bool(paddle.isfinite(x).all().numpy()):
            bad.append(name)

            if name.endswith('_mean') or name.endswith('_variance'):
                bad_bn.append(name)
            else:
                bad_non_bn.append(name)

    print(
        os.path.basename(p),
        '| bad =', len(bad),
        '| bad_bn =', len(bad_bn),
        '| bad_non_bn =', len(bad_non_bn)
    )

    if bad:
        print('  BAD:')
        for n in bad[:20]:
            print('   ', n)

print('=============================================')
