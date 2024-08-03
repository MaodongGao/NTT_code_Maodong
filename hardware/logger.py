from loguru import logger
import os, sys, zmq, inspect, json
from zmq.log.handlers import PUBHandler
import time
logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[devicename]}</cyan> | "
    "<cyan>{extra[function_module]}</cyan>:<cyan>{extra[function_name]}</cyan>:<cyan>{extra[function_line]}</cyan>\n"
    "<level>{message}</level>")

tcp_address = "tcp://127.0.0.1:12345"


def get_call_kwargs(level=1):
    frame = inspect.currentframe().f_back
    for _ in range(level):  # get the level-th frame
        frame = frame.f_back
    code_obj = frame.f_code
    return dict(
        function_module=os.path.basename(code_obj.co_filename),
        function_name=code_obj.co_name,
        function_line=frame.f_lineno,
    )



class LoggerClient:
    def __init__(self, init_logger=True) -> None:
        self.tcp_address = tcp_address
        if init_logger:
            self.__init_logger()

    def __init_logger(self):
        logger.remove()
        logger.add(sys.stderr, format=logger_format, level="DEBUG")  # Add all levels to console

        context = zmq.Context.instance()
        self.socket = context.socket(zmq.PUB)
        self.socket.connect(self.tcp_address)
        self.info("Logger client started at " + self.tcp_address, name="LoggerClient", **get_call_kwargs(level=0))

    def log(self, level, msg, name='', *args, **kwargs):
        # Log locally with the specified level
        log_kwargs = {**get_call_kwargs(level=2), 'devicename': name, **kwargs}
        getattr(logger.bind(**log_kwargs), level.lower())(msg)

        # Send serialized log record over ZMQ
        serialized_record = json.dumps({'level': level, 'msg': msg, 'kwargs': log_kwargs}).encode('utf-8')
        self.socket.send_multipart([b"log", serialized_record])

    def debug(self, msg, name='', *args, **kwargs):
        self.log('DEBUG', msg, name, *args, **kwargs)

    def info(self, msg, name='', *args, **kwargs):
        self.log('INFO', msg, name, *args, **kwargs)

    def warning(self, msg, name='', *args, **kwargs):
        self.log('WARNING', msg, name, *args, **kwargs)

    def error(self, msg, name='', *args, **kwargs):
        self.log('ERROR', msg, name, *args, **kwargs)

    def critical(self, msg, name='', *args, **kwargs):
        self.log('CRITICAL', msg, name, *args, **kwargs)


class LoggerServer:
    def __init__(self) -> None:
        self.tcp_address = tcp_address
        self.logfile = os.path.join(os.path.expanduser('~'), 'Desktop', 'Logs', 'test_server.log')
        self.email_recipient = "maodong.gao@ntt-research.com"  # email address to send log file when it is rotated

    def start(self):
        self.start_server()

    def start_server(self):
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.SUB)
        self.socket.bind(self.tcp_address)
        self.socket.subscribe("")
        logger.remove()
        logger.add(sys.stderr, format=logger_format, level="DEBUG")  # Add all levels to console
        logger.add(self.logfile, format=logger_format, level="DEBUG", rotation="1 kB", retention=5, enqueue=True, compression=self.__send_log_file_via_email)

        logger.bind(devicename="LoggerServer").info("Logger server started at " + self.tcp_address, **get_call_kwargs(level=0))
        
        while True:
            try:
                _, message = self.socket.recv_multipart()
                log_record = json.loads(message.decode("utf8").strip())

                # Log with extra fields and correct level
                getattr(logger.bind(**log_record['kwargs']), log_record['level'].lower())(log_record['msg'])

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error occurred: {e}")
        
    def __send_log_file_via_email(self, log_file_path):
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.application import MIMEApplication
        mail_content = "Please find the attached log file."
        subject = "Log File Notification"
        recv_address = self.email_recipient

        sender_address = 'maodongntt@gmail.com'
        sender_pass = 'dsyt xhmu rsjk vntj'

        to_string = ', '.join(recv_address)

        # Message initialization
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = to_string
        message['Subject'] = subject

        # Attach the log file
        if os.path.isfile(log_file_path):
            with open(log_file_path, 'rb') as file:
                part = MIMEApplication(file.read(), Name=os.path.basename(log_file_path))
            # After the file is closed
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(log_file_path)}"'
            message.attach(part)

        # Attach the mail content
        message.attach(MIMEText(mail_content, 'plain'))

        # SMTP session
        try:
            session = smtplib.SMTP('smtp.gmail.com', 587)
            session.starttls()
            session.login(sender_address, sender_pass)
            text = message.as_string()
            session.sendmail(sender_address, recv_address, text)
            print(f"Sent log file to {recv_address} successfully.")
        except Exception as e:
            print(f"Failed to send email: {e}")
        finally:
            session.quit()

        print("send log file via email FUNCTION CALLED")
        pass





## Deprecated
# class LoggerClient:
#     def __init__(self, init_logger=True) -> None:
#         self.tcp_address = tcp_address
#         if init_logger:
#             self.__init_logger()

#     def __init_logger(self):
#         logger.remove()
#         logger.add(sys.stderr, format=logger_format, level="INFO") # recover console print

#         context = zmq.Context.instance()
#         self.socket = context.socket(zmq.PUB)
#         self.socket.connect(self.tcp_address)
#         handler = PUBHandler(self.socket)
#         logger.add(handler)

#         logger.bind(devicename="LoggerClient").info("Logger client started at " + self.tcp_address, **get_call_kwargs(level=0))

#     def info(self, msg, name='', *args, **kwargs):
#         logger.info(msg, devicename=name, *args, **get_call_kwargs(level=1), **kwargs)



# class LoggerServer:
#     def __init__(self) -> None:
#         self.tcp_address = tcp_address
#         self.logfile = os.path.expanduser(r'~\Desktop\Logs\test_server.log')
#         self.email_recipient = "" # email address to send log file wwhen it is rotated

#     def start(self):
#         self.start_server()

#     def start_server(self):
#         self.socket = zmq.Context().socket(zmq.SUB)
#         self.socket.bind(self.tcp_address)
#         self.socket.subscribe("")
#         logger.remove()  
#         logger.add(sys.stderr, format=logger_format, level="INFO") # recover console print
#         logger.add(self.logfile, format=logger_format, level="INFO", rotation="1 kB", retention=5, enqueue=True, compression=self.__send_log_file_via_email)  # 1MB per file, 5 files max
#         logger.bind(devicename="LoggerServer").info("Logger server started at " + self.tcp_address, **get_call_kwargs(level=0))
#         while True:
#             try:
#                 _, message = self.socket.recv_multipart()
#                 logger.info(message.decode("utf8").strip())
#             except KeyboardInterrupt:
#                 break
#             except Exception as e:
#                 continue

#     def __send_log_file_via_email(self):
#         print("send log file via email FUNCTION CALLED")
#         pass
