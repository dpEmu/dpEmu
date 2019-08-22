import numpy as np
from dpemu.nodes import Array
from dpemu.filters import Constant, Addition, Subtraction, Multiplication, Division, IntegerDivision, Identity
from dpemu.filters import Min, Max, Difference, Modulo, And, Or, Xor


def test_constant():
    a = np.arange(25).reshape((5, 5))
    params = {'c': 5}
    x_node = Array()
    x_node.addfilter(Constant('c'))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 5))


def test_identity():
    a = np.arange(25).reshape((5, 5))
    x_node = Array()
    x_node.addfilter(Identity())
    out = x_node.generate_error(a, {}, np.random.RandomState(seed=42))
    assert np.array_equal(out, a)


def test_addition():
    a = np.full((5, 5), 5)
    params = {}
    params['c'] = 2
    x_node = Array()
    x_node.addfilter(Addition(Constant('c'), Identity()))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 7))


def test_subtraction():
    a = np.full((5, 5), 5)
    params = {}
    params['c'] = 2
    x_node = Array()
    x_node.addfilter(Subtraction(Constant('c'), Identity()))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), -3))


def test_multiplication():
    a = np.full((5, 5), 5)
    params = {}
    params['c'] = 2
    x_node = Array()
    x_node.addfilter(Multiplication(Constant('c'), Identity()))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 10))


def test_division():
    a = np.full((5, 5), 5.0)
    params = {}
    params['c'] = 2
    x_node = Array()
    x_node.addfilter(Division(Constant('c'), Identity()))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.allclose(out, np.full((5, 5), .4))


def test_integer_division():
    a = np.full((5, 5), 5)
    params = {}
    params['c'] = 2
    x_node = Array()
    x_node.addfilter(IntegerDivision(Identity(), Constant('c')))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 2))


def test_modulo():
    a = np.full((5, 5), 5)
    params = {}
    params['c'] = 2
    x_node = Array()
    x_node.addfilter(Modulo(Identity(), Constant('c')))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 1))


def test_and():
    a = np.full((5, 5), 5)
    params = {}
    params['c'] = 2
    x_node = Array()
    x_node.addfilter(And(Identity(), Constant('c')))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 0))


def test_or():
    a = np.full((5, 5), 5)
    params = {}
    params['c'] = 2
    x_node = Array()
    x_node.addfilter(Or(Identity(), Constant('c')))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 7))


def test_xor():
    a = np.full((5, 5), 5)
    params = {}
    params['c'] = 3
    x_node = Array()
    x_node.addfilter(Xor(Identity(), Constant('c')))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 6))


def test_difference():
    a = np.full((5, 5), 5)
    params = {}
    params['c'] = 2
    addition = Addition(Identity(), Constant('c'))
    x_node = Array()
    x_node.addfilter(Difference(addition))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))

    assert np.array_equal(out, np.full((5, 5), 2))


def test_min():
    a = np.full((5, 5), 5)
    params = {}
    params['c'] = 2
    x_node = Array()
    x_node.addfilter(Min(Identity(), Constant('c')))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 2))


def test_max():
    a = np.full((5, 5), 5)
    params = {}
    params['c'] = 2
    x_node = Array()
    x_node.addfilter(Max(Identity(), Constant('c')))
    out = x_node.generate_error(a, params, np.random.RandomState(seed=42))
    assert np.array_equal(out, np.full((5, 5), 5))
