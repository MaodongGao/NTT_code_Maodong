from loguru import logger
import os, sys, zmq, inspect, json, warnings
import multiprocessing
logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[devicename]}</cyan> | "
    "procID <cyan>{extra[process_id]}</cyan> | "
    "<cyan>{extra[function_module]}</cyan>:<cyan>{extra[function_name]}</cyan>:<cyan>{extra[function_line]}</cyan>\n"
    "<level>{message}</level>")

tcp_address = "tcp://127.0.0.1:12345"

def get_call_kwargs(level=1):
    '''
    Get the function name, line number, and module name of the caller function
    
    Parameters:
        level (int): the level of the caller function, default is 1
        
    Returns:
        dict: a dictionary containing the function name, line number, and module name of the caller function
    '''
    frame = inspect.currentframe().f_back
    for _ in range(level):  # get the level-th frame
        frame = frame.f_back
    code_obj = frame.f_code
    return dict(
        function_module=os.path.basename(code_obj.co_filename),
        function_name=code_obj.co_name,
        function_line=frame.f_lineno,
        process_id=str(os.getpid()),
    )

def send_email(subject, mail_content, recv_address, attachment=None, priority="Normal", is_html=False):
    '''
    Send email with attachment
    
    Parameters:
        subject (str): email subject
        mail_content (str): email content
        recv_address (list): list of email addresses to receive the email, e.g. ["recipient1@example.com", "recipient2@example.com"]
        attachment (str): path to the attachment file, e.g. "C:/Users/username/Desktop/attachment.txt"
        priority (str): email priority, e.g. "Normal", "Urgent", "Non-Urgent"
        is_html (bool): whether the email content is HTML formatted

    '''
    import smtplib
    import os
    import warnings
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication

    sender_address = 'maodongntt@gmail.com'
    sender_pass = 'dsyt xhmu rsjk vntj'

    to_string = ', '.join(recv_address)

    # Message initialization
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = to_string
    message['Subject'] = subject

    # Set email priority
    if priority.lower() == "urgent":
        message['X-Priority'] = '1'  # Urgent
    elif priority.lower() == "non-urgent":
        message['X-Priority'] = '5'  # Non-Urgent
    else:
        message['X-Priority'] = '3'  # Normal

    # Attach the attachment
    if attachment is not None:
        if os.path.isfile(attachment):
            with open(attachment, 'rb') as file:
                part = MIMEApplication(file.read(), Name=os.path.basename(attachment))
            # After the file is closed
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment)}"'
            message.attach(part)
        else:
            warnings.warn(f"Attachment file {attachment} does not exist. Email sent without attachment.")

    # Attach the mail content
    if is_html:
        message.attach(MIMEText(mail_content, 'html'))
    else:
        message.attach(MIMEText(mail_content, 'plain'))

    # SMTP session
    try:
        session = smtplib.SMTP('smtp.gmail.com', 587)
        session.starttls()
        session.login(sender_address, sender_pass)
        text = message.as_string()
        session.sendmail(sender_address, recv_address, text)
        print(f"Sent email to {recv_address} successfully.")
    # except Exception as e:
    #     print(f"Failed to send email: {e}")
    finally:
        session.quit()
    print("send email FUNCTION CALLED")

class LoggerClient:
    '''
    Logger client class to send log
    
    Parameters:
        init_logger (bool): whether to initialize the logger client, default is True
        
    Attributes:
        tcp_address (str): the tcp address of the logger server
        critical_warning_recipient (list): email address to send when a CRITICAL log is received
        socket (zmq.Socket): the ZMQ socket to send log

    Example Usage:
        logger = LoggerClient()
        logger.critical_warning_recipient =  ["recipient1@example.com", "recipient2@example.com"]
        logger.info("This is an info log", name="Device1") # log an info message with device name "Device1"
        logger.critical("This is a critical log", name="Device2") # Critical message will be logged and an Email will be sent to the critical_warning_recipient

    Main Features:
        1. Print log information to console, with additional information including device name, process ID, function name, and line number
        2. When a CRITICAL log is received, an email will be sent to the critical_warning_recipient
        3. Log info is also sent to tcp_address via ZMQ
        4. When LoggerServer is also running, LoggerServer will receive the log at tcp_address and log it to a file
        5. LoggerClient can be instantiated in multiple python processes, and all logs will be sent to the same tcp_address to handle log files in a single LoggerServer process
    '''
    def __init__(self, init_logger=True) -> None:
        self.tcp_address = tcp_address
        self.critical_warning_recipient =  [
            "maodong.gao@ntt-research.com", # always blocked? 
            "maodonggao@outlook.com", # Need to add to white list
            "gaomaodong@126.com"
            ] # email address to send when a CRITICAL log is received
        if init_logger:
            self.__init_logger()

    def __init_logger(self):
        logger.remove()
        
        # logger.add(sys.stderr, format=logger_format, level="DEBUG")  # Add all levels to console

        context = zmq.Context.instance()
        self.socket = context.socket(zmq.PUB)
        self.socket.connect(self.tcp_address)
        import time
        time.sleep(0.1)  # Wait for the connection to be established
        self.info("Logger client started at " + self.tcp_address, name="LoggerClient", **get_call_kwargs(level=0))

    def log(self, level, msg, name='', *args, **kwargs):
        if msg is None:
            # print("msg is None, no log will be recorded.")
            return
        
        # Add support to log ndarray
        flag = False
        if 'ndarray' in kwargs:
            flag = True
            import numpy as np
            array = np.array(kwargs.pop('ndarray'))
            if 'ndarray_precision' in kwargs:
                precision = kwargs.pop('ndarray_precision')
            else:
                precision = None
            with np.printoptions(threshold=np.inf, suppress=True, precision=precision):
                msg = f"{array}\n{msg}"
                
        # Log locally with the specified level
        log_kwargs = {**get_call_kwargs(level=2), 'devicename': name, **kwargs}
        getattr(logger.bind(**log_kwargs), level.lower())(msg)

        # Send serialized log record over ZMQ
        serialized_record = json.dumps({'level': level, 'msg': msg, 'kwargs': log_kwargs}).encode('utf-8')
        self.socket.send_multipart([b"log", serialized_record])

        # Add support to log ndarray
        if flag:
            self.info(f"ndarray with type {array.dtype} and shape {array.shape} taking {len(msg.encode('utf-8'))/1024:.3f} kB memory is logged.", name=name, **log_kwargs)

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
        self.__send_critical_email(msg, name, *args, **kwargs)

    def __send_critical_email(self, msg, name='', *args, **kwargs):
        import platform
        laptop_name = platform.node()
        subject = "[CRITICAL][NTT-PHI-Lab] Critical warning on " + laptop_name
        import time
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mail_content = f"""
        <html>
        <body>
            <p>[Automatically Generated Email]</p>
            <p>Hello,</p>
            <p>A CRITICAL warning was received from computer: <strong>{laptop_name}</strong> at <strong>{current_time}, {time.strftime('%Z')}</strong>. Please check your setup IMMEDIATELY.</p>
            <p><b><font color="red">The CRITICAL warning message is:</font></b> {msg}</p>
            <p>Device name: {name}</p>
        """
        trigger_info = get_call_kwargs(level=2)
        mail_content += f"""
            <p>Triggered by: {trigger_info['function_module']}:{trigger_info['function_name']}:{str(trigger_info['function_line'])}</p>
            <p>Process ID: {trigger_info['process_id']}</p>
            <p>This email is for CRITICAL warning notification only.</p>
            <p>Best regards,<br>Maodong</p>
        </body>
        </html>
        """

        recv_address = self.critical_warning_recipient

        try:
            send_email(subject, mail_content, recv_address, priority="Urgent", is_html=True)
            logger.bind(devicename="LoggerClient").info(f"Critical warning email sent to {recv_address}", **get_call_kwargs(level=0))
        except Exception as e:
            logger.bind(devicename="LoggerClient").error(f"Critical warning email failed to send to {recv_address}. Error: {e}.", **get_call_kwargs(level=0))







class LoggerServer:
    '''
    Logger server class to receive log
    
    Attributes:
        tcp_address (str): the tcp address of the logger server
        logfile (str): the log file path
        email_recipient (list): email address to send log file when it is rotated
        socket (zmq.Socket): the ZMQ socket to receive log
        
    Example Usage:
        # You should always have a LoggerServer process (separate from LoggerClient processes) running to handle recording logs to file and email backup
        LoggerServer().start() 
        # start the LoggerServer process (This process should be a single process running in the background, separate from LoggerCLient processes)

    Main Features:
        1. Receive log from LoggerClient via ZMQ at tcp_address, print the received log to server console, and log it to a log file.
        2. When the log file size exceeds predefined size, the log file will be rotated and sent to email_recipient as an attachment (as a backup). 
        3. Locally, the log file only retains 5 newest files.
        4. The log file rotation is handled within a single LoggerServer process, and multiple LoggerClient processes can send log to the same tcp_address. 
           Therefore the previous bug that rotation process can't rotate the log file when multiple LoggerClient processes are running is fixed.
    '''
    def __init__(self) -> None:
        self.tcp_address = tcp_address
        self.logfile = os.path.join(os.path.expanduser('~'), 'Desktop', 'Logs', 'test_server.log')
        self.email_recipient = [
            "maodong.gao@ntt-research.com", # always blocked? 
            "maodonggao@outlook.com", # Need to add to white list
            "gaomaodong@126.com"
            ] # email address to send log file when it is rotated

    def start(self):
        self.start_server()

    def start_server(self):
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.SUB)
        self.socket.bind(self.tcp_address)
        self.socket.subscribe("")
        logger.remove()
        logger.add(sys.stderr, format=logger_format, level="DEBUG")  # Add all levels to console
        logger.add(self.logfile, format=logger_format, level="DEBUG", rotation="10 MB", retention=5, enqueue=True, compression=self.__send_log_file_via_email)

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
        import platform
        laptop_name = platform.node()
        subject = "[Regular][NTT-PHI-Lab] log file rotated on " + laptop_name + ": " + str(os.path.split(log_file_path)[1])
        mail_content = '[Automatically Generated Email] \n\n' + 'Hello, \n\n Attached are the rotated data logging file at NTT-PHI-Lab. This log file is sent from computer: ' + laptop_name + '. \n\n This email is for log file backup purpose only. \n\n Best, \n Maodong'
        recv_address = self.email_recipient

        try:
            send_email(subject, mail_content, recv_address, attachment=log_file_path, is_html=False)
            logger.bind(devicename="LoggerServer").info(f"Rotated Log file {log_file_path} on {laptop_name} sent to {recv_address}", **get_call_kwargs(level=0))
        except Exception as e:
            logger.bind(devicename="LoggerServer").error(f"Rotated Log file {log_file_path} on {laptop_name} failed to send to {recv_address}. Error: {e}.", **get_call_kwargs(level=0))


    # def __send_log_file_via_email(self, log_file_path):
    #     # Start a new process to send the email
    #     process = multiprocessing.Process(target=self.__send_email_process, args=(log_file_path,))
    #     process.start()

    # def __send_email_process(self, log_file_path):
    #     import platform
    #     laptop_name = platform.node()
    #     subject = "[Regular][NTT-PHI-Lab] log file rotated on " + laptop_name + ": " + str(os.path.split(log_file_path)[1])
    #     mail_content = '[Automatically Generated Email] \n\n' + 'Hello, \n\n Attached are the rotated data logging file at NTT-PHI-Lab. This log file is sent from computer: ' + laptop_name + '. \n\n This email is for log file backup purpose only. \n\n Best, \n Maodong'
    #     recv_address = self.email_recipient

    #     try:
    #         send_email(subject, mail_content, recv_address, attachment=log_file_path, is_html=False)
    #         logger.bind(devicename="LoggerServer").info(f"Rotated Log file {log_file_path} on {laptop_name} sent to {recv_address}", **get_call_kwargs(level=0))
    #     except Exception as e:
    #         logger.bind(devicename="LoggerServer").error(f"Rotated Log file {log_file_path} on {laptop_name} failed to send to {recv_address}. Error: {e}.", **get_call_kwargs(level=0))







## Deprecated

# def send_email(subject, mail_content, recv_address, attachment=None):
#     '''
#     Send email with attachment
    
#     Parameters:
#         subject (str): email subject
#         mail_content (str): email content in plain text
#         recv_address (list): list of email addresses to receive the email, e.g. ["recipient1@example.com", "recipient2@example.com", "recipient3@example.com"]
#         attachment (str): path to the attachment file, e.g. "C:/Users/username/Desktop/attachment.txt"

#     '''
#     import smtplib
#     from email.mime.multipart import MIMEMultipart
#     from email.mime.text import MIMEText
#     from email.mime.application import MIMEApplication
#     sender_address = 'maodongntt@gmail.com'
#     sender_pass = 'dsyt xhmu rsjk vntj'

#     to_string = ', '.join(recv_address)

#     # Message initialization
#     message = MIMEMultipart()
#     message['From'] = sender_address
#     message['To'] = to_string
#     message['Subject'] = subject

#     # Attach the attachment
#     if attachment is not None:
#         if os.path.isfile(attachment):
#             with open(attachment, 'rb') as file:
#                 part = MIMEApplication(file.read(), Name=os.path.basename(attachment))
#             # After the file is closed
#             part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment)}"'
#             message.attach(part)
#         else:
#             warnings.warn(f"Attachment file {attachment} does not exist. Email sent without attachment.")

#     # Attach the mail content
#     message.attach(MIMEText(mail_content, 'plain'))

#     # SMTP session
#     try:
#         session = smtplib.SMTP('smtp.gmail.com', 587)
#         session.starttls()
#         session.login(sender_address, sender_pass)
#         text = message.as_string()
#         session.sendmail(sender_address, recv_address, text)
#         print(f"Sent email to {recv_address} successfully.")
#     except Exception as e:
#         print(f"Failed to send email: {e}")
#     finally:
#         session.quit()
#     print("send email FUNCTION CALLED")
# 
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



    # def __send_log_file_via_email(self, log_file_path):
    #     # Start a new process to send the email
    #     process = multiprocessing.Process(target=self.__send_email_process, args=(log_file_path,))
    #     process.start()

    # def __send_email_process(self, log_file_path):
    #     import platform
    #     laptop_name = platform.node()
    #     subject = "[Regular][NTT-PHI-Lab] log file rotated on " + laptop_name + ": " + str(os.path.split(log_file_path)[1])
    #     mail_content = '[Automatically Generated Email] \n\n' + 'Hello, \n\n Attached are the rotated data logging file at NTT-PHI-Lab. This log file is sent from computer: ' + laptop_name + '. \n\n This email is for log file backup purpose only. \n\n Best, \n Maodong'
    #     recv_address = self.email_recipient

    #     try:
    #         send_email(subject, mail_content, recv_address, attachment=log_file_path, is_html=False)
    #         logger.bind(devicename="LoggerServer").info(f"Rotated Log file {log_file_path} on {laptop_name} sent to {recv_address}", **get_call_kwargs(level=0))
    #     except Exception as e:
    #         logger.bind(devicename="LoggerServer").error(f"Rotated Log file {log_file_path} on {laptop_name} failed to send to {recv_address}. Error: {e}.", **get_call_kwargs(level=0))
