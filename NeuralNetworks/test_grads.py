def numerical_grad(f, arr, eps=1e-6):
    """Finite-difference gradient of scalar f() w.r.t. arr, perturbing arr in place."""
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        i = it.multi_index
        old = arr[i]
        arr[i] = old + eps; plus = f()
        arr[i] = old - eps; minus = f()
        arr[i] = old
        grad[i] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad

def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))

def grad_check_layer(layer, x_shape, seed=0):
    r_gen = np.random.default_rng(seed)
    x = r_gen.standard_normal(x_shape)
    r = r_gen.standard_normal(layer.forward(x).shape)   # fixed random "upstream gradient"

    def loss():
        return np.sum(layer.forward(x) * r)

    layer.forward(x)
    dx = layer.backprop(r)
    analytic = [g.copy() for g in layer.grads()]        # copy before the numeric pass overwrites state

    print(type(layer).__name__)
    print("  input grad error:", rel_error(dx, numerical_grad(loss, x)))
    for k, p in enumerate(layer.params()):
        print(f"  param {k} grad error:", rel_error(analytic[k], numerical_grad(loss, p)))

def grad_check_loss(loss_layer, pred_shape, target, seed=0):
    pred = np.random.default_rng(seed).standard_normal(pred_shape)
    loss_layer.forward(pred, target)
    analytic = loss_layer.backprop()
    numeric = numerical_grad(lambda: loss_layer.forward(pred, target), pred)
    print(type(loss_layer).__name__)
    print("  loss grad error:", rel_error(analytic, numeric))