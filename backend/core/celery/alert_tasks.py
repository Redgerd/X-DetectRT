from celery import shared_task
from config import settings
import json
import redis
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging

# Redis client for pub/sub
redis_client = redis.Redis(host="redis", port="6379", db=0, decode_responses=True)

@shared_task(name="core.celery.alert_tasks.publish_alert")
def publish_alert(alert_data: dict):
    """
    Publishes an alert to the Redis Pub/Sub channel for real-time updates.
    """
    camera_id = alert_data.get("camera_id")
    if not camera_id:
        return {"error": "Camera ID is required"}

    channel = f"camera_alerts:{camera_id}"  # Format Redis channel name
    redis_client.publish(channel, json.dumps(alert_data))  # Publish to Redis
    
    return {"status": "Alert published successfully"}


@shared_task(name="core.celery.alert_tasks.send_email")
def send_email(sender_email:str, sender_password:str, receiver_emails:str, subject:str, body:str, server:str, port:int=465):
    """
    Sends an email using the SMTP protocol.

    Args:
        sender_email (str): The email address of the sender.
        sender_password (str): The password of the sender's email.
        receiver_emails (str): The comma-separated list of email addresses of the receivers.
        subject (str): The subject of the email.
        body (str): The body of the email.
        server (str): The SMTP server address.
        port (int): The port number of the SMTP server.
    """
    try:
        server = smtplib.SMTP_SSL(server, port)
        server.login(sender_email, sender_password)

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_emails
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        text = msg.as_string()
        server.sendmail(from_addr=sender_email, to_addrs=receiver_emails.split(","), msg=text)
        logging.info("Email sent successfully.")
        server.quit()

        return {"status": "Email sent successfully"}
    
    except Exception as e:
        logging.exception(e)