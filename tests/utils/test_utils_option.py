from ivy.utils import Option

from unittest import TestCase


class TestOption(TestCase):
    "Testing Class for Option type class"

    def setUp(self):
        self.option_some_int = Option[int](3)
        self.option_none = Option[int](None)

    def test_option_some_unwrap(self):
        self.assertEqual(self.option_some_int.unwrap(), 3)
    
    def test_option_construct_some(self):
        option: Option[int] = Option.some(3)
        self.assertEqual(option.unwrap(), 3)
        self.assertEqual(self.option_some_int.unwrap(), option.unwrap())

    def test_option_none_unwrap(self):
        self.assertIsNone(self.option_none.unwrap())
    
    def test_option_construct_none(self):
        option: Option[int] = Option.none()
        self.assertIsNone(option.unwrap())

    def test_option_none_isnone(self):
        self.assertTrue(self.option_none.is_none())

    def test_option_some_isnone(self):
        self.assertFalse(self.option_some_int.is_none())
