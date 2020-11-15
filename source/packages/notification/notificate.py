import smtplib


def sendemail(subject, message, from_addr='jan.hruska.test@gmail.com', to_addr_list=['14mato41@gmail.com'], cc_addr_list=['jan.hruska.test@gmail.com'],
              login='jan.hruska.test@gmail.com', password='@r%%^/M+fwE{h~3p',
              smtpserver='smtp.gmail.com:587'):
    header = 'From: %s' % from_addr
    header += '\nTo: %s' % ','.join(to_addr_list)
    header += '\nCc: %s' % ','.join(cc_addr_list)
    header += '\nSubject: %s' % subject
    message = header + "\n"+message

    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login, password)
    problems = server.sendmail(from_addr, to_addr_list, message)
    server.quit()
