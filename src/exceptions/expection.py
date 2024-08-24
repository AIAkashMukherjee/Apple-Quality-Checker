import sys

def error_msg_detail(error,error_detail:sys):
    _,_,ex_tb=error_detail.exc_info()

    file_name=ex_tb.tb_frame.f_code.co_filename

    error_msg="Error occured in python script name [{0}] line number [{1}] erro message [{2}]".format(
        file_name, ex_tb.tb_lineno, str(error)
    )

    return error_msg


class CustomException(Exception):
    def __init__(self,error_msg,error_detail):
        super().__init__(error_msg)
        self.error_message=error_msg_detail(error_msg,error_detail)

    def __str__(self) -> str:
        return self.error_message