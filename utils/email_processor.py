"""
Email Processing Module
Handles various email file formats including .eml, .msg, and .pst files
"""
import re
import email
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Email processing libraries
try:
    import extract_msg
    EXTRACT_MSG_AVAILABLE = True
except ImportError:
    EXTRACT_MSG_AVAILABLE = False
    logging.warning("extract-msg not available. .msg files will not be supported.")

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    logging.warning("BeautifulSoup not available. HTML email parsing will be limited.")

logger = logging.getLogger(__name__)

class EmailProcessor:
    """Process various email file formats and extract text content"""
    
    def __init__(self):
        self.supported_formats = ['.eml', '.msg']
        if EXTRACT_MSG_AVAILABLE:
            self.supported_formats.append('.msg')
    
    def is_email_file(self, file_path: str) -> bool:
        """Check if file is an email format"""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_formats
    
    def get_email_info(self, file_path: str) -> Dict:
        """Get basic information about the email file"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.eml':
            return self._get_eml_info(file_path)
        elif ext == '.msg' and EXTRACT_MSG_AVAILABLE:
            return self._get_msg_info(file_path)
        else:
            return {"error": f"Unsupported email format: {ext}"}
    
    def _get_eml_info(self, file_path: str) -> Dict:
        """Extract information from .eml file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                msg = email.message_from_file(f)
            
            return {
                "subject": msg.get('subject', 'No Subject'),
                "from": msg.get('from', 'Unknown Sender'),
                "to": msg.get('to', 'Unknown Recipient'),
                "date": msg.get('date', 'Unknown Date'),
                "content_type": msg.get_content_type(),
                "has_attachments": self._has_attachments(msg)
            }
        except Exception as e:
            logger.error(f"Error reading EML file {file_path}: {e}")
            return {"error": str(e)}
    
    def _get_msg_info(self, file_path: str) -> Dict:
        """Extract information from .msg file"""
        try:
            msg = extract_msg.Message(file_path)
            return {
                "subject": msg.subject or 'No Subject',
                "from": msg.sender or 'Unknown Sender',
                "to": msg.to or 'Unknown Recipient',
                "date": str(msg.date) if msg.date else 'Unknown Date',
                "content_type": "application/vnd.ms-outlook",
                "has_attachments": bool(msg.attachments)
            }
        except Exception as e:
            logger.error(f"Error reading MSG file {file_path}: {e}")
            return {"error": str(e)}
    
    def _has_attachments(self, msg: email.message.Message) -> bool:
        """Check if email has attachments"""
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue
            return True
        return False
    
    def extract_email_content(self, file_path: str) -> str:
        """Extract text content from email file"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.eml':
            return self._extract_eml_content(file_path)
        elif ext == '.msg' and EXTRACT_MSG_AVAILABLE:
            return self._extract_msg_content(file_path)
        else:
            return f"Error: Unsupported email format: {ext}"
    
    def _extract_eml_content(self, file_path: str) -> str:
        """Extract content from .eml file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                msg = email.message_from_file(f)
            
            content_parts = []
            
            # Add email headers
            headers = [
                f"Subject: {msg.get('subject', 'No Subject')}",
                f"From: {msg.get('from', 'Unknown Sender')}",
                f"To: {msg.get('to', 'Unknown Recipient')}",
                f"Date: {msg.get('date', 'Unknown Date')}",
                f"Content-Type: {msg.get_content_type()}",
                ""
            ]
            content_parts.extend(headers)
            
            # Extract body content
            body_content = self._extract_email_body(msg)
            if body_content:
                content_parts.append("BODY:")
                content_parts.append(body_content)
                content_parts.append("")
            
            # Extract attachments info
            attachments = self._extract_attachments_info(msg)
            if attachments:
                content_parts.append("ATTACHMENTS:")
                content_parts.extend(attachments)
            
            return "\n".join(content_parts)
            
        except Exception as e:
            logger.error(f"Error extracting EML content from {file_path}: {e}")
            return f"Error extracting email content: {str(e)}"
    
    def _extract_msg_content(self, file_path: str) -> str:
        """Extract content from .msg file"""
        try:
            msg = extract_msg.Message(file_path)
            content_parts = []
            
            # Add email headers
            headers = [
                f"Subject: {msg.subject or 'No Subject'}",
                f"From: {msg.sender or 'Unknown Sender'}",
                f"To: {msg.to or 'Unknown Recipient'}",
                f"Date: {str(msg.date) if msg.date else 'Unknown Date'}",
                f"Content-Type: application/vnd.ms-outlook",
                ""
            ]
            content_parts.extend(headers)
            
            # Extract body content
            if msg.body:
                content_parts.append("BODY:")
                content_parts.append(msg.body)
                content_parts.append("")
            
            # Extract attachments info
            if msg.attachments:
                content_parts.append("ATTACHMENTS:")
                for attachment in msg.attachments:
                    content_parts.append(f"- {attachment.longFilename or attachment.shortFilename}")
            
            return "\n".join(content_parts)
            
        except Exception as e:
            logger.error(f"Error extracting MSG content from {file_path}: {e}")
            return f"Error extracting email content: {str(e)}"
    
    def _extract_email_body(self, msg: email.message.Message) -> str:
        """Extract body content from email message"""
        body_parts = []
        
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            
            content_type = part.get_content_type()
            content_disposition = part.get('Content-Disposition', '')
            
            # Skip attachments
            if 'attachment' in content_disposition.lower():
                continue
            
            # Handle text content
            if content_type == 'text/plain':
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        text = payload.decode(charset, errors='ignore')
                        body_parts.append(text)
                except Exception as e:
                    logger.warning(f"Error decoding text/plain part: {e}")
            
            # Handle HTML content
            elif content_type == 'text/html' and BEAUTIFULSOUP_AVAILABLE:
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        html = payload.decode(charset, errors='ignore')
                        # Convert HTML to text
                        soup = BeautifulSoup(html, 'html.parser')
                        text = soup.get_text(separator='\n', strip=True)
                        body_parts.append(text)
                except Exception as e:
                    logger.warning(f"Error decoding text/html part: {e}")
        
        return "\n\n".join(body_parts)
    
    def _extract_attachments_info(self, msg: email.message.Message) -> List[str]:
        """Extract information about attachments"""
        attachments = []
        
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            
            content_disposition = part.get('Content-Disposition', '')
            if 'attachment' in content_disposition.lower():
                filename = part.get_filename()
                content_type = part.get_content_type()
                if filename:
                    attachments.append(f"- {filename} ({content_type})")
        
        return attachments
    
    def process_email_file(self, file_path: str, output_dir: str = None) -> str:
        """Process email file and return markdown content"""
        if not self.is_email_file(file_path):
            return f"Error: {file_path} is not a supported email format"
        
        try:
            # Extract email content
            content = self.extract_email_content(file_path)
            
            # Create markdown file
            if output_dir is None:
                output_dir = os.path.dirname(file_path)
            
            filename = Path(file_path).stem
            markdown_path = os.path.join(output_dir, f"{filename}_email.md")
            
            # Write markdown content
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(f"# Email: {filename}\n\n")
                f.write(content)
            
            logger.info(f"Email processed successfully: {markdown_path}")
            return markdown_path
            
        except Exception as e:
            logger.error(f"Error processing email file {file_path}: {e}")
            return f"Error processing email: {str(e)}"

# Global email processor instance
email_processor = EmailProcessor()

def is_email_file(file_path: str) -> bool:
    """Check if file is an email format"""
    return email_processor.is_email_file(file_path)

def process_email_file(file_path: str, output_dir: str = None) -> str:
    """Process email file and return markdown file path"""
    return email_processor.process_email_file(file_path, output_dir)

def get_email_info(file_path: str) -> Dict:
    """Get email file information"""
    return email_processor.get_email_info(file_path) 