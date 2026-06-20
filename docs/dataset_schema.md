# 📦 Dataset Schema – ATO Detection

## Processed Dataset: `data/ato_dataset_processed.csv`

| Column | Type | Description |
|--------|------|-------------|
| transaction_id | str | Unique transaction ID |
| account_number | str | Customer account number |
| transaction_timestamp | datetime | UTC timestamp |
| transaction_amount | float | Amount in INR |
| transaction_type | str | NEFT/IMPS/UPI/RTGS |
| ip_address | str | Client IP address |
| ip_country | str | ISO country code |
| ip_city | str | City from IP geolocation |
| **ip_blacklisted** | bool | IP in blacklist |
| **ip_fraud_score** | float [0-1] | IP reputation score |
| device_id | str | Device fingerprint |
| device_type | str | Android/iOS/Windows |
| device_os | str | OS version |
| **device_blacklisted** | bool | Device in blacklist |
| **new_device** | bool | Not seen before |
| location_city | str | User's location city |
| location_state | str | User's location state |
| **new_location** | bool | Different from usual |
| **location_mismatch_ip** | bool | Location ≠ IP location |
| auth_method | str | PASSWORD/PIN |
| **failed_auth_attempts** | int | Failed logins this session |
| **session_age_minutes** | int | Session duration to auth |
| beneficiary_account | str | Destination account |
| beneficiary_name | str | Destination name |
| **new_beneficiary** | bool | First-time recipient |
| **beneficiary_blacklisted** | bool | Recipient in blacklist |
| channel | str | mobile/web |
| **txn_frequency_1h** | int | Transactions in last hour |
| **txn_frequency_24h** | int | Transactions in last day |
| **amount_vs_avg** | float | Ratio to historical average |
| **amount_percentile** | float [0-1] | Historical percentile |
| **account_age_days** | int | Account age |
| **previous_fraud_flags** | int | Past fraud incidents |
| kyc_status | str | VERIFIED/PENDING |
| risk_profile | str | LOW/MEDIUM/HIGH |
| **is_ato** | bool | **Target label** |
| **typing_speed_mobile** | float | Keys/second on mobile |
| **typing_rhythm_mobile** | float | Keystroke timing variance |
| **phone_angle_pitch** | float | Device tilt (0-90°) |
| **phone_angle_roll** | float | Device roll angle |
| **swipe_speed_mobile** | float | Swipe velocity |
| **mouse_path_complexity** | float | Mouse path fractal dim. |
| **typing_speed_web** | float | Keys/second on web |
| **typing_rhythm_web** | float | Web keystroke timing |
| **mouse_speed_web** | float | Mouse pixels/second |
| **scroll_velocity_mobile** | float | Scroll speed mobile |
| **scroll_velocity_web** | float | Scroll speed web |
| **session_duration** | float | Session length (seconds) |
| **behavioral_anomaly_score** | float [0-1] | Composite behavioral score |

**Bold** = used as model features.
