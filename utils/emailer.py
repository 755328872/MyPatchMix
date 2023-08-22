import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import os
import smtplib
import traceback
from .logger import Logger

from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from tensorboard.backend.event_processing import event_accumulator


class EmailNotifier:
    def __init__(self, sender_email, sender_pwd, receiver_email):
        self.emailer = None
        self.sender_email = sender_email
        self.sender_pwd = sender_pwd
        self.receiver_email = receiver_email

    def __enter__(self):
        if self.emailer == None:
            self.emailer = Emailer()
        # 检查必要信息
        if (not self.sender_email) or (not self.sender_pwd) or (not self.receiver_email):
            raise Exception("必要信息缺失，请重新输入发件人的邮箱和授权码，收件人的邮箱")
        self.emailer.setSender(self.sender_email, self.sender_pwd)
        self.emailer.setReceiver(self.receiver_email)
        return self.emailer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type or exc_val or exc_tb:
            dict1 = {
                "异常类型": str(exc_type),
                "异常值": str(exc_val),
                "异常回溯信息": str(exc_tb),
                "详细报错信息": str(traceback.format_exc())
            }
            self.emailer.update_Dict(dict1)
            self.emailer.send_email(self.emailer.return_text_dict, run_success=False)
        else:
            self.emailer.send_email(self.emailer.return_text_dict, run_success=True)

class Emailer:
    def __init__(self, logger_tb=None, email_type='QQ'):
        # 'QQ' / '163'
        self.email_type = email_type
        # 发送邮箱，可以用我的，也可以自己搞个授权码
        self.sender_email = '755328872@qq.com'
        self.sender_password = 'ptrubfrjrippbbbe'  # QQ邮箱密码是生成的授权码
        # 收件邮箱
        self.receiver_email = '755328872@qq.com'
        # 邮件标题
        self.subject = 'Experiment result'
        # 邮件正文内容
        self.main_content = 'Experiment results:'
        # 日志管理器
        self.logger = logger_tb
        # 邮件发送器
        self.return_text_dict = {}

        #=====================================
        # ```
        # Emailer类功能为邮件提醒。
        # 使用前提：
        # 1. 需要将日志文件和events文件保存在./log文件夹下，不然读取不好读
        # 2. 需要输出的必要信息可以通过类方法update_Dict传递
        # 3. 图片命名不要以events和console开头
        # ```
        #=====================================

    # 更新字典到return_text_dict
    def update_Dict(self, dict={}):
        self.return_text_dict.update(dict)

    def send_email(self, res_text={}, run_success=True):
        # 构建主题
        msg = MIMEMultipart()

        # 邮件内容(最开始的main方法——try catch来决定发不发报错信息)
        if run_success:
            # 构造正文
            body_content = self.main_content + '\n' + "Congratulation! Your program has succeeded!" + '\n'

            # 提取res_text字典
            for k, v in res_text.items():
                body_content += "<h3>" + str(k) + ":</h3> " + "\n" + str(v) + "\n"
            msg.attach(MIMEText(body_content, 'html', 'utf-8'))  # 构建邮件正文,不能多次构造

            # 查找events文件中的数据, 直接从logger读取路径
            src_dir = self.logger.log_dir
            fnames = os.listdir(src_dir)
            events_name = ''
            console_log_name = ''
            for f in fnames:
                # ...最后改回去
                if f.startswith("events.out.tfevents"):
                    events_name = f
                if f.startswith("console"):
                    console_log_name = f

            # 事件文件所在路径
            event_dir = os.path.join(src_dir, events_name)
            # 日志文件所在路径
            console_log_dir =  os.path.join(src_dir, console_log_name)
            ea = event_accumulator.EventAccumulator(event_dir)
            ea.Reload()

            # 根据events数据绘制并保存图片
            photos_filenames = []
            for key in ea.scalars.Keys():
                datas_labels = ea.scalars.Items(key)  # list
                plt.plot([i.step for i in datas_labels],[i.value for i in datas_labels], label=str(key))
                # 得到label
                if '/' in key:
                    xlabel = key.split('/')[0]
                    ylabel = key.split('/')[1]
                else:
                    xlabel = 'x_label'
                    ylabel = 'y_label'
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.legend()
                photo_name = xlabel + '_' + ylabel + '.png'
                plt.savefig(photo_name)
                photos_filenames.append(photo_name)
                plt.close()

            # 添加图片附件
            for filename in photos_filenames:
                with open(filename, 'rb') as f:
                    attachment = MIMEImage(f.read(), name=os.path.basename(filename))
                    msg.attach(attachment)

            # 添加日志附件
            with open(console_log_dir, 'rb') as f:
                attachment = MIMEText(f.read(), "plain", "utf-8")
                attachment["Content-Disposition"] = "attachment; filename=" + console_log_name
                msg.attach(attachment)
        else:
            self.subject = "喜报！程序寄了"
            body_content = "<h2>" + "程序运行报错了，内容如下：" + "</h2>" + "\n"
            # 提取res_text字典
            for k, v in res_text.items():
                body_content += "<h3>" + str(k) + ":</h3> " + "\n" + str(v) + "\n"

            msg.attach(MIMEText(body_content, 'html', 'utf-8'))  # 构建邮件正文,不能多次构造

        # ===================== 发送邮件 ====================
        if self.email_type == 'QQ':
            # QQ邮箱作为发送方，需要授权码
            mail_host = 'smtp.qq.com'
            port = 465
        elif self.email_type == '163':
            # 163邮箱作为发送方，需要授权码
            mail_host = 'smtp.163.com'
            port = 465
        elif self.email_type == '126':
            # 126邮箱作为发送方，需要授权码
            mail_host = 'smtp.126.com'
            port = 465

        msg['Subject'] = Header(self.subject,'utf-8')  # 邮件主题
        msg['From'] = self.sender_email  # 发件人
        msg['To'] = self.receiver_email  # 收件人

        server = smtplib.SMTP_SSL(mail_host, port)
        server.set_debuglevel(1)
        server.login(self.sender_email, self.sender_password)
        server.sendmail(self.sender_email, self.receiver_email, msg.as_string())
        server.quit()

    def _setEmailType(self, type_str):
        self.email_type = type_str

    def setSender(self, email, password):
        assert isinstance(email, str) and isinstance(password, str)
        self.sender_email = email
        self.sender_password = password
        if "@QQ" in self.sender_email:
            self._setEmailType("QQ")
        elif "@163" in self.sender_email:
            self._setEmailType("163")
        elif "@126" in self.sender_email:
            self._setEmailType("126")

    def setReceiver(self, email):
        assert isinstance(email, str)
        self.receiver_email = email

    def setSubject(self, subject):
        assert isinstance(subject, str)
        self.subject = subject

    def setMainContent(self, content):
        assert isinstance(content, str)
        self.main_content = content

    def setLogger(self, logger):
        assert isinstance(logger, Logger)
        self.logger = logger

