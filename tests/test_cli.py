import contextlib
import io
import unittest

from vramfit.cli import main


class CliTest(unittest.TestCase):
    def test_no_arguments_prints_help(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("usage: vramfit", output.getvalue())


if __name__ == "__main__":
    unittest.main()
