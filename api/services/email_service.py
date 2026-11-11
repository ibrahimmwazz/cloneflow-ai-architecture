"""
Email service for sending transactional emails using Resend API.
Supports welcome emails, receipts, and alerts.
"""

import os
from typing import Optional, Dict, Any
from datetime import datetime
import httpx
from pydantic import BaseModel, EmailStr

from api.config import settings


class EmailTemplate(BaseModel):
    """Base email template model"""
    to_email: EmailStr
    subject: str
    html_content: str
    text_content: Optional[str] = None


class EmailService:
    """Service for sending transactional emails via Resend API"""
    
    def __init__(self):
        self.api_key = settings.RESEND_API_KEY
        self.from_email = settings.EMAIL_FROM_ADDRESS or "noreply@cloneflow.ai"
        self.from_name = settings.EMAIL_FROM_NAME or "CloneFlow AI"
        self.base_url = "https://api.resend.com/emails"
    
    async def send_email(self, template: EmailTemplate) -> bool:
        """Send an email using Resend API"""
        if not self.api_key:
            print("Email service not configured (RESEND_API_KEY not set)")
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "from": f"{self.from_name} <{self.from_email}>",
                "to": [template.to_email],
                "subject": template.subject,
                "html": template.html_content,
            }
            
            if template.text_content:
                payload["text"] = template.text_content
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.base_url, headers=headers, json=payload)
                response.raise_for_status()
                return True
                
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False
    
    async def send_welcome_email(self, to_email: str, user_name: Optional[str] = None) -> bool:
        """Send welcome email to new user"""
        subject = "Welcome to CloneFlow AI! 🎉"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome to CloneFlow AI</title>
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px;">
                <h1 style="color: white; margin: 0; font-size: 28px;">Welcome to CloneFlow AI!</h1>
                <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">Your AI video journey starts now</p>
            </div>
            
            <p style="font-size: 16px;">Hi {user_name or 'there'},</p>
            
            <p style="font-size: 16px;">Welcome aboard! We're thrilled to have you join CloneFlow AI. You now have access to the most advanced AI avatar video platform.</p>
            
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h2 style="color: #667eea; margin-top: 0;">🎁 Your Free Credits</h2>
                <p style="font-size: 16px;">You've been credited with <strong>10 free credits</strong> to start creating amazing avatar videos!</p>
            </div>
            
            <h2 style="color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px;">Getting Started</h2>
            <ol style="font-size: 16px;">
                <li style="margin-bottom: 10px;"><strong>Upload a portrait</strong> - Upload a photo or generate an AI avatar</li>
                <li style="margin-bottom: 10px;"><strong>Write your script</strong> - Type what you want to say or use AI to generate it</li>
                <li style="margin-bottom: 10px;"><strong>Choose a voice</strong> - Select from presets or clone your own voice</li>
                <li style="margin-bottom: 10px;"><strong>Generate</strong> - Create your video in ~90 seconds</li>
            </ol>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="https://cloneflow.ai/lab" style="display: inline-block; background: #667eea; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px;">Create Your First Video →</a>
            </div>
            
            <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0;">
                <p style="margin: 0; font-size: 14px;"><strong>💡 Pro Tip:</strong> Check out our <a href="https://cloneflow.ai/help" style="color: #667eea;">Help Center</a> for tutorials and guides!</p>
            </div>
            
            <p style="font-size: 16px;">If you have any questions, just reply to this email. We're here to help!</p>
            
            <p style="font-size: 16px;">Happy creating!</p>
            
            <p style="font-size: 16px; margin-bottom: 30px;">The CloneFlow AI Team</p>
            
            <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
            
            <p style="font-size: 12px; color: #999; text-align: center;">
                You received this email because you signed up for CloneFlow AI.<br>
                <a href="https://cloneflow.ai/unsubscribe" style="color: #999;">Unsubscribe</a>
            </p>
        </body>
        </html>
        """
        
        template = EmailTemplate(
            to_email=to_email,
            subject=subject,
            html_content=html_content
        )
        
        return await self.send_email(template)
    
    async def send_receipt_email(
        self,
        to_email: str,
        user_name: Optional[str],
        amount: float,
        credits: int,
        transaction_id: str,
        date: datetime
    ) -> bool:
        """Send purchase receipt email"""
        subject = f"Receipt: {credits} Credits Purchase"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Purchase Receipt</title>
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px;">
                <h1 style="color: white; margin: 0; font-size: 28px;">Payment Successful! ✅</h1>
                <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">Your credits have been added</p>
            </div>
            
            <p style="font-size: 16px;">Hi {user_name or 'there'},</p>
            
            <p style="font-size: 16px;">Thank you for your purchase! Your credits have been successfully added to your account.</p>
            
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h2 style="color: #333; margin-top: 0;">Receipt Details</h2>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 10px 0; border-bottom: 1px solid #ddd;"><strong>Transaction ID:</strong></td>
                        <td style="padding: 10px 0; border-bottom: 1px solid #ddd;">{transaction_id}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 0; border-bottom: 1px solid #ddd;"><strong>Date:</strong></td>
                        <td style="padding: 10px 0; border-bottom: 1px solid #ddd;">{date.strftime('%B %d, %Y at %I:%M %p')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 0; border-bottom: 1px solid #ddd;"><strong>Credits Purchased:</strong></td>
                        <td style="padding: 10px 0; border-bottom: 1px solid #ddd;"><strong>{credits} credits</strong></td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 0;"><strong>Amount Paid:</strong></td>
                        <td style="padding: 10px 0;"><strong>${amount:.2f} USD</strong></td>
                    </tr>
                </table>
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="https://cloneflow.ai/dashboard" style="display: inline-block; background: #667eea; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px;">Go to Dashboard →</a>
            </div>
            
            <p style="font-size: 16px;">Start creating amazing avatar videos with your new credits!</p>
            
            <p style="font-size: 16px; margin-bottom: 30px;">The CloneFlow AI Team</p>
            
            <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
            
            <p style="font-size: 12px; color: #999; text-align: center;">
                This is an automatic receipt for your purchase.<br>
                <a href="https://cloneflow.ai/support" style="color: #999;">Need help? Contact Support</a>
            </p>
        </body>
        </html>
        """
        
        template = EmailTemplate(
            to_email=to_email,
            subject=subject,
            html_content=html_content
        )
        
        return await self.send_email(template)
    
    async def send_alert_email(
        self,
        to_email: str,
        alert_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send alert email for important notifications"""
        subject = f"CloneFlow Alert: {alert_type}"
        
        metadata_html = ""
        if metadata:
            metadata_html = "<ul style='font-size: 14px;'>"
            for key, value in metadata.items():
                metadata_html += f"<li><strong>{key}:</strong> {value}</li>"
            metadata_html += "</ul>"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Alert</title>
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 20px; border-radius: 8px; margin-bottom: 30px;">
                <h1 style="color: #856404; margin: 0 0 10px 0; font-size: 24px;">⚠️ {alert_type}</h1>
                <p style="margin: 0; font-size: 16px;">{message}</p>
            </div>
            
            {metadata_html and f"<div style='background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;'><h3 style='color: #333; margin-top: 0;'>Details</h3>{metadata_html}</div>"}
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="https://cloneflow.ai/dashboard" style="display: inline-block; background: #667eea; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px;">Go to Dashboard →</a>
            </div>
            
            <p style="font-size: 16px; margin-bottom: 30px;">The CloneFlow AI Team</p>
            
            <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
            
            <p style="font-size: 12px; color: #999; text-align: center;">
                This is an automated alert from CloneFlow AI.<br>
                <a href="https://cloneflow.ai/support" style="color: #999;">Need help? Contact Support</a>
            </p>
        </body>
        </html>
        """
        
        template = EmailTemplate(
            to_email=to_email,
            subject=subject,
            html_content=html_content
        )
        
        return await self.send_email(template)
    
    async def send_video_ready_email(
        self,
        to_email: str,
        user_name: Optional[str],
        video_url: str,
        job_id: str
    ) -> bool:
        """Send notification when video is ready"""
        subject = "Your Avatar Video is Ready! 🎬"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Video Ready</title>
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px;">
                <h1 style="color: white; margin: 0; font-size: 28px;">Your Video is Ready! 🎬</h1>
                <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">It looks amazing!</p>
            </div>
            
            <p style="font-size: 16px;">Hi {user_name or 'there'},</p>
            
            <p style="font-size: 16px;">Great news! Your avatar video has been successfully generated and is ready for download.</p>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="{video_url}" style="display: inline-block; background: #667eea; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px;">Download Video →</a>
            </div>
            
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #333; margin-top: 0;">Video Details</h3>
                <p style="font-size: 14px; margin: 5px 0;"><strong>Job ID:</strong> {job_id}</p>
                <p style="font-size: 14px; margin: 5px 0;"><strong>Status:</strong> Completed</p>
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="https://cloneflow.ai/dashboard" style="display: inline-block; background: white; color: #667eea; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px; border: 2px solid #667eea;">View in Dashboard →</a>
            </div>
            
            <p style="font-size: 16px;">Share your video with the world! Don't forget to tag us @cloneflowai</p>
            
            <p style="font-size: 16px; margin-bottom: 30px;">The CloneFlow AI Team</p>
            
            <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
            
            <p style="font-size: 12px; color: #999; text-align: center;">
                This is an automated notification from CloneFlow AI.<br>
                <a href="https://cloneflow.ai/unsubscribe" style="color: #999;">Unsubscribe</a>
            </p>
        </body>
        </html>
        """
        
        template = EmailTemplate(
            to_email=to_email,
            subject=subject,
            html_content=html_content
        )
        
        return await self.send_email(template)


# Singleton instance
email_service = EmailService()
