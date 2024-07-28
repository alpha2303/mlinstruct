from ivy.utils import Option

from unittest import TestCase


class TestOption(TestCase):
    "Testing Class for Option type class"

    def setUp(self) -> None:
        self.option_some_int = Option[int](3)
        self.option_none = Option[int](None)

    def test_option_some_unwrap(self) -> None:
        self.assertEqual(self.option_some_int.unwrap(), 3)

    def test_option_construct_some(self) -> None:
        option: Option[int] = Option.some(3)
        self.assertEqual(option.unwrap(), 3)
        self.assertEqual(self.option_some_int.unwrap(), option.unwrap())

    def test_option_none_unwrap(self) -> None:
        self.assertIsNone(self.option_none.unwrap())

    def test_option_construct_none(self) -> None:
        option: Option[int] = Option.none()
        self.assertIsNone(option.unwrap())

    def test_option_none_isnone(self) -> None:
        self.assertTrue(self.option_none.is_none())

    def test_option_some_isnone(self) -> None:
        self.assertFalse(self.option_some_int.is_none())
