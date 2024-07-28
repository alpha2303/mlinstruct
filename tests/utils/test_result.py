from ivy.utils import Result

from unittest import TestCase


class TestResult(TestCase):
    "Testing Class for Result type class"

    def setUp(self) -> None:
        self.result_ok = Result[int, ZeroDivisionError](5)
        self.result_err = Result[int, ZeroDivisionError](err=ZeroDivisionError())

    def test_result_ok_unwrap(self) -> None:
        self.assertEqual(self.result_ok.unwrap(), 5)

    def test_result_construct_ok(self) -> None:
        result: Result[int, ZeroDivisionError] = Result.ok(5)
        self.assertEqual(result.unwrap(), 5)
        self.assertEqual(result.unwrap(), self.result_ok.unwrap())

    def test_result_err_unwrap(self) -> None:
        self.assertTrue(isinstance(self.result_err.unwrap(), ZeroDivisionError))

    def test_result_constructor_err(self) -> None:
        result: Result[int, ZeroDivisionError] = Result.err(ZeroDivisionError())
        self.assertTrue(result.is_err())
        self.assertTrue(isinstance(result.unwrap(), ZeroDivisionError))
        self.assertEqual(type(result.unwrap()), type(self.result_err.unwrap()))
        self.assertEqual(result.unwrap().args, self.result_err.unwrap().args)

    def test_result_err_iserr(self) -> None:
        self.assertTrue(self.result_err.is_err())

    def test_result_ok_iserr(self) -> None:
        self.assertFalse(self.result_ok.is_err())
