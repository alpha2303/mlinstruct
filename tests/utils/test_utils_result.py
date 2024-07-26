from ivy.utils import Result

from unittest import TestCase
import pytest


class TestResult(TestCase):
    "Testing Class for Option type class"

    def setUp(self):
        self.result_ok = Result[int, ZeroDivisionError](5)
        self.result_err = Result[int, ZeroDivisionError](err=ZeroDivisionError())

    def test_result_ok_unwrap(self):
        self.assertEqual(self.result_ok.unwrap(), 5)
    
    def test_result_construct_ok(self):
        result: Result[int, ZeroDivisionError] = Result.ok(5)
        self.assertEqual(result.unwrap(), 5)
        self.assertEqual(result.unwrap(), self.result_ok.unwrap())

    def test_result_err_unwrap(self):
        self.assertTrue(isinstance(self.result_err.unwrap(), ZeroDivisionError))
    
    def test_result_constructor_err(self):
        result: Result[int, ZeroDivisionError] = Result.err(ZeroDivisionError())
        self.assertTrue(result.is_err())
        self.assertTrue(isinstance(result.unwrap(), ZeroDivisionError))
        self.assertEqual(type(result.unwrap()), type(self.result_err.unwrap()))
        self.assertEqual(result.unwrap().args, self.result_err.unwrap().args)

    def test_result_err_iserr(self):
        self.assertTrue(self.result_err.is_err())

    def test_result_ok_iserr(self):
        self.assertFalse(self.result_ok.is_err())
