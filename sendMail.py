import smtplib

from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders

fromaddr = "trung.tran@smartcube.vn"  # your email
toaddr = "trung.tranviet94@gmail.com" # receiver 
msg = MIMEMultipart()    
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "Checking auto email"  # SUBJECT of email
body = "Done checking, you can do it now"  # Body of email
try:
  msg.attach(MIMEText(body, 'plain'))
  filename = "file.log"  # file's dir
  attachment = open(filename, "rb")
  part = MIMEBase('application', 'octet-stream')
  part.set_payload((attachment).read())
  encoders.encode_base64(part)
  part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
  msg.attach(part)
  server = smtplib.SMTP('smtp.gmail.com', 587)
  server.starttls()
  server.login(fromaddr, "pemmjtsabviyndih")  # Password application
  text = msg.as_string()
  server.sendmail(fromaddr, toaddr, text)
  server.quit()
except Exception as e:
  print(str(e))