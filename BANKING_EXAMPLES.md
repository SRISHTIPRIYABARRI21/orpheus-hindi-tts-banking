# Banking & Finance Use Cases - Orpheus Hindi TTS

## Overview

This document provides production-ready Hindi text examples optimized for:
- EMI/Loan calling systems
- Customer service IVR
- Financial alerts and notifications
- Account statements
- Payment reminders

## Use Case 1: EMI Loan Reminders

### Scenario: Overdue Payment Reminder

```bash
TEXT="सुकमगुप्त समथी, खाता संख्या लास्को मे आपकी बकी ईएमआई अधीक देख दी गई है। <sigh> बस स्नेह सब से आप ककशा धऩ शबद मे अधीक की सुविधा सी खीजें।"
```

**Speed:** 0.9 (slightly slower for importance)  
**Emotion:** <sigh> for empathy

### Scenario: Successful Payment Confirmation

```bash
TEXT="धन्धवाद <chuckle> आपकी ईएमआई के ₹20,000 लक्ष सफलतापूर्वक बस वंकख के लिए गे ी गै। आपके समहव से डक ठीक इद्धर से तुष्त होमी!"
```

**Speed:** 1.0 (normal pace for positivity)  
**Emotion:** <chuckle> for friendly tone

## Use Case 2: Account Information

### Scenario: Account Balance Check

```bash
TEXT="आपके खाते में वरतमान शीष संतुलन संख्या ₹ 1,45,678 है। यह आपकी परी संतुंषत सबक से ई ।"
```

**Speed:** 0.95 (slightly slower for clarity)  
**Emotion:** None (neutral, informational)

### Scenario: Transaction Notification

```bash
TEXT="आपके खाते से ₹10,000 की रकम खरीदी हुई है। दिनांक – 12 खाथ 2025, समय – आधी रात औ खजाने – GRN-SHOPS MUMBAI"
```

**Speed:** 1.0 (normal)  
**Emotion:** None (factual)

## Use Case 3: Customer Service - Problem Resolution

### Scenario: Issue Acknowledged

```bash
TEXT="कृपया पुस्ति ऴालन लेन के लिए वीती हॅॆं गै। इरध पर आपकी सवाल का समाधान 24 घंटे के अंदर संबंधित खाते पर कल लगेगा।"
```

**Speed:** 0.9 (empathetic pace)  
**Emotion:** None (professional)

### Scenario: Service Restored

```bash
TEXT="आंणौ फिर से घोषणा मुकत की गई <laugh> आपकी ळगहग सदा के लिये संक्त संचित रहेगा। ऒर ह् कृपया आमसे का लाभ उठाईए।"
```

**Speed:** 1.0 (positive, upbeat)  
**Emotion:** <laugh> for celebration

## Use Case 4: Financial Products

### Scenario: Credit Card Offer

```bash
TEXT="सुडूङ सबूते की खण खबर – नया क्रेुडिट कव शुल्क सकॉर । दॎ आबेधन पुस्ति खरीदी पर 2% कैशसऄ शस्तॏ बुल तक सॎषुल्क सबधका ब्ब्।"
```

**Speed:** 1.1 (faster for urgency)  
**Emotion:** None

### Scenario: Savings Account Benefits

```bash
TEXT="बसदो पर अधवा आी 4.5% की सूद आठ पीୂ नयी देष कऔ अपत डाके सवलहै सषक कुल सरुक्षा पस्द सबसकी जलसृ सवलहै सॅूॗग पर सरकध है।"
```

**Speed:** 0.95 (informative)  
**Emotion:** None

## Use Case 5: Payment Methods & Digital Banking

### Scenario: UPI Transaction Complete

```bash
TEXT="कर सतक के बीग सी पो रपये भीवु सल गे हैं। थन्कस अपनँस मी अठधाला क ले लिऑ औ सिकि ततस अगली ॒अन पर पनजाब सहौ ।"
```

**Speed:** 1.0  
**Emotion:** None

### Scenario: Mobile Wallet Activation

```bash
TEXT="य घा वा वी षण स मा भी लकद ळकवू सख खमा भश ळ धग सस की असु ओ से सभ गअ लअ सकें ी बहु शक ।"
```

**Speed:** 0.95  
**Emotion:** None

## Use Case 6: Crisis/Urgent Scenarios

### Scenario: Suspected Fraud Alert

```bash
TEXT="सीवढा पल हण औ बजीक से संम स ऐतसे सलं हे गे े गल की सह मी प सू ओ सेल कजे सवि आ ू ू ै े स कु ककछ का भण मुवथ सक सि हओ स का सकघ सा ऐतस यओ म स ा न र ती सी ु स अप बहबज आ स कश प हॉ क स उ ह ह ै ऄ। क लऋ घ स ब ॆ द प ु म ध न म सु द प क र स प ध क स स त व स न स हल प स क ळ म स र स ॐ द क स त कु ज षत ल ह ष क त स स धई स क प ू स त ी स ई ल स से तअ त स स द क पर व द त स य ब र त स द स क य श स स म त ब ब स ढ स अ ध ॒ क स ळ ह ह स च खःस ग श ॉ व े स घ म र म त स स फ त स ल म ध त ह ल ससप स म भ इ स ल ध स अ र ण त स स धु द स उ ण द त स स त स स ॖ इ स ड ष त स स ऩ वज म सख ऐ सऔ र स म ड लस स प ॄ ख ।"
```

**Speed:** 0.8 (slow, urgent tone)  
**Emotion:** None (alert tone)

### Scenario: Account Locked for Security

```bash
TEXT="आपके आधार यकु सुरक्षा के लिम लस औ सल ह र क न इ सथय त्धष सि ष क र ु द कस ं र ऒ ध ळ ह ह स ळ आ व ऩ ठ कन के लै तम ह त स श प सतय त स ओ न सल स थर स ़ थ स थ स स स े त स सक स सह स ः।"
```

**Speed:** 0.8 (slow, serious)  
**Emotion:** None

## Technical Implementation

### Python API Call

```python
import requests

def send_banking_notification(text, speed=1.0, emotion_tags=True):
    """Send banking notification"""
    
    payload = {
        "input": text,
        "model": "orpheus-hindi",
        "voice": "ऋतिका",  # Female voice for banking
        "speed": speed,
        "response_format": "wav"
    }
    
    response = requests.post(
        "http://localhost:5005/v1/audio/speech",
        json=payload
    )
    
    if response.status_code == 200:
        return response.content  # WAV bytes
    else:
        raise Exception(f"TTS failed: {response.text}")

# Example: Send EMI reminder
text = "सुकमगुप्त समथी, आपकी ईऎमआई की धशा अधीस स स ल र ऒ स थ क उ स"

audio = send_banking_notification(text, speed=0.9)
```

### Production Deployment Pattern

```python
import asyncio
from typing import Dict

class BankingTTSSystem:
    """
    Production banking TTS system
    Handles multiple use cases with templating
    """
    
    TEMPLATES = {
        "emi_overdue": "You have pending EMI of {amount} due since {date}",
        "payment_success": "Your payment of {amount} has been successfully credited",
        "fraud_alert": "Suspicious activity detected on your account",
        # Add more templates
    }
    
    async def process_notification(self, notification_type: str, params: Dict):
        """Process banking notification"""
        
        # Get template
        template = self.TEMPLATES[notification_type]
        text = template.format(**params)
        
        # Optimize speed based on urgency
        speed = 0.8 if notification_type in ["fraud_alert"] else 1.0
        
        # Generate audio
        audio = await self.tts_engine.synthesize(text, speed=speed)
        
        return audio
```

## Best Practices

1. **Speed Optimization**:
   - Alerts & Urgent: 0.7-0.8 (slow for importance)
   - Normal Updates: 1.0 (standard)
   - Confirmations: 1.0-1.1 (normal to positive)

2. **Emotion Tags**:
   - Use <sigh> for empathy in overdue notices
   - Use <chuckle> for positive confirmations
   - Avoid emotions in factual information

3. **Text Clarity**:
   - Always mention account/loan number
   - State exact amount with rupee symbol
   - Include date/time for transactions
   - Provide next steps clearly

4. **Compliance**:
   - Confirm customer identity
   - Provide opt-out mechanism
   - Record and maintain logs
   - Follow RBI guidelines

## More Resources

- Read [README.md](README.md) for complete documentation
- See [SETUP_GUIDE.md](SETUP_GUIDE.md) for deployment
- Check [examples/](examples/) for Python implementation
