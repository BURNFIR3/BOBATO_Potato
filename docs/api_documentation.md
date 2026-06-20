# 📡 API Documentation – Bank of Baroda ATO Detection

Base URL: `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

---

## 1. Detect ATO

**POST** `/api/v1/detect-ato`

Analyze a transaction for account takeover risk.

### Request Body
```json
{
  "account_number": "BOB12345678",
  "transaction_amount": 50000.0,
  "transaction_type": "NEFT",
  "ip_address": "196.6.8.10",
  "ip_fraud_score": 0.95,
  "ip_country": "NG",
  "new_device": true,
  "new_location": true,
  "new_beneficiary": true,
  "failed_auth_attempts": 4,
  "typing_speed_mobile": 8.5,
  "phone_angle_pitch": 10.0,
  "behavioral_anomaly_score": 0.75
}
```

### Response
```json
{
  "transaction_id": "TXN-20240620-ABC123",
  "account_number": "BOB12345678",
  "ato_probability": 0.9200,
  "behavioral_anomaly_score": 0.8100,
  "action": "SUSPEND",
  "suspension_mode": "REVIEW",
  "otp_required": true,
  "contact_mobile": true,
  "behavioral_warning": "High behavioral anomaly: 0.810 (threshold: 0.30)",
  "user_capabilities": {
    "view_balance": true,
    "view_history": true,
    "call_bank": true,
    "transfer_money": false,
    "add_beneficiary": false,
    "change_data": false
  },
  "user_actions": [
    {"action": "SUSPEND_ACCOUNT", "description": "Account suspended for security review"},
    {"action": "TRIGGER_OTP", "description": "OTP sent to registered mobile", "otp_method": "SMS"},
    {"action": "CONTACT_REGISTERED_MOBILE", "description": "SMS: Suspicious activity detected"}
  ],
  "blacklist_result": {
    "ip_blacklisted": true,
    "device_blacklisted": true,
    "beneficiary_blacklisted": true,
    "reason": "ATO fraud confirmed"
  },
  "timestamp": "2024-06-20T12:00:00.000000"
}
```

---

## 2. Verify OTP

**POST** `/api/v1/verify-otp`

```json
{
  "account_number": "BOB12345678",
  "otp": "482916"
}
```

### Response (success)
```json
{
  "account_number": "BOB12345678",
  "otp_valid": true,
  "action": "ACCOUNT_UNLOCKED",
  "message": "Identity verified. Account unlocked."
}
```

### Response (failure)
```json
{
  "account_number": "BOB12345678",
  "otp_valid": false,
  "action": "ACCOUNT_BLOCKED",
  "message": "OTP verification failed. Account fully blocked. KYC re-check required."
}
```

---

## 3. Statistics

**GET** `/api/v1/ato-stats`

**GET** `/api/v1/fraud-transactions`

**GET** `/api/v1/suspended-accounts`

**GET** `/api/v1/health`

---

## 4. Blacklists

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/blacklist/ip` | Get full IP blacklist |
| POST | `/api/v1/blacklist/ip` | Add IP |
| DELETE | `/api/v1/blacklist/ip/{ip}` | Remove IP |
| GET | `/api/v1/blacklist/ip/{ip}/check` | Check if blacklisted |
| GET | `/api/v1/blacklist/device` | Get device blacklist |
| POST | `/api/v1/blacklist/device` | Add device |
| GET | `/api/v1/blacklist/beneficiary` | Get beneficiary blacklist |
| GET | `/api/v1/blacklist/summary` | Counts across all lists |

---

## 5. Behavioral Baseline

**POST** `/api/v1/behavioral/baseline/{account_number}`

```json
{
  "sessions": [
    {"typing_speed_mobile": 4.5, "phone_angle_pitch": 45.0, ...},
    {"typing_speed_mobile": 4.3, "phone_angle_pitch": 44.0, ...},
    {"typing_speed_mobile": 4.7, "phone_angle_pitch": 46.0, ...}
  ]
}
```

**GET** `/api/v1/behavioral/baseline/{account_number}`

**POST** `/api/v1/behavioral/anomaly-check`
