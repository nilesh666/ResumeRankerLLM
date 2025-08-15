import sys

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    return f"Error in this {filename} file, in the line {exc_tb.tb_lineno} and the error is {str(error)}"

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message= error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message