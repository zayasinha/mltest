import sys
from src.mltest.logger import logging


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    line_number = exc_tb.tb_lineno
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script: {file_name} at line number: {line_number} with message: {str(error)}"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail=None):
        # allow callers to pass either (error) or (error, sys)
        super().__init__(str(error_message))
        self.error_message = CustomException.get_detailed_error_message(error_message, error_detail)

    def __str__(self):
        return self.error_message

    @staticmethod
    def get_detailed_error_message(error, error_detail=None):
        """Return a detailed error string using the provided error_detail (usually the sys module).

        If error_detail is None, the global sys module will be used.
        """
        if error_detail is None:
            error_detail = sys
        try:
            return error_message_detail(error, error_detail)
        except Exception:
            # Fallback: log and return a basic string representation
            logging.error("Failed to get detailed error message", exc_info=True)
            return str(error)