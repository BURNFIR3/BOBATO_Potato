"""
Generate the synthetic behavioral ATO dataset from the pasted specification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import RAW_DATA_DIR


def generate_behavioral_dataset(
    n: int = 50_000,
    fraud_rate: float = 0.08,
    noise_rate: float = 0.15,
    label_noise_rate: float = 0.01,
    output_path: str | Path | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic ATO behavioral dataset and save it as CSV."""
    rng = np.random.default_rng(random_state)
    fraud_bool = rng.binomial(1, fraud_rate, n)

    is_known_vpn = np.where(
        fraud_bool == 1,
        rng.binomial(1, 0.75, n),
        rng.binomial(1, 0.02, n),
    )
    is_foreign_ip = np.where(
        (fraud_bool == 1) & (is_known_vpn == 1),
        rng.binomial(1, 0.92, n),
        np.where(
            (fraud_bool == 1) & (is_known_vpn == 0),
            rng.binomial(1, 0.30, n),
            rng.binomial(1, 0.01, n),
        ),
    )
    country_mismatch = np.where(is_foreign_ip == 1, rng.binomial(1, 0.88, n), 0)
    ip_distance = np.where(
        fraud_bool == 1,
        rng.uniform(2000, 12000, n),
        rng.exponential(30, n),
    )
    days_since_last = np.where(
        fraud_bool == 1,
        rng.uniform(0.01, 0.5, n),
        rng.lognormal(1.5, 0.8, n),
    )
    speed = ip_distance / (days_since_last * 24)

    is_new_device = np.where(
        fraud_bool == 1,
        rng.binomial(1, 0.95, n),
        rng.binomial(1, 0.12, n),
    )
    canvas = [f"{rng.integers(0, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64):016x}" for _ in range(n)]
    webgl = [f"{rng.integers(0, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64):016x}" for _ in range(n)]
    ua_raw = np.where(
        fraud_bool == 1,
        rng.choice(
            [
                "Mozilla/5.0 (X11; Linux x86_64)",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "Mozilla/5.0 (Android 13; Mobile)",
            ],
            n,
        ),
        rng.choice(
            [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)",
                "Mozilla/5.0 (Android 12; Mobile) AppleWebKit/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
            ],
            n,
        ),
    )
    network_type = np.where(
        is_known_vpn == 1,
        "VPN",
        rng.choice(["Cable/DSL", "Mobile", "Datacenter"], n, p=[0.7, 0.2, 0.1]),
    )
    device_os = np.where(
        fraud_bool == 1,
        rng.choice(["Windows", "Linux", "Android", "iOS"], n, p=[0.4, 0.3, 0.2, 0.1]),
        rng.choice(["Windows", "Android", "iOS", "MacOS"], n, p=[0.5, 0.25, 0.2, 0.05]),
    )
    source = np.where(fraud_bool == 1, rng.choice(["web", "api"], n, p=[0.7, 0.3]), "web")

    login_hour = np.where(fraud_bool == 1, rng.choice([2, 3, 4, 5, 23, 0], n), rng.integers(8, 23, n))
    anomaly_score = np.where(fraud_bool == 1, rng.beta(2, 0.5, n), rng.beta(0.5, 2, n))
    mouse_vel = np.where(
        fraud_bool == 1,
        rng.choice([0, 5, 10], n, p=[0.5, 0.3, 0.2]),
        rng.normal(220, 80, n).clip(0),
    )
    mobile_os = np.isin(device_os, ["Android", "iOS"])
    touch_radius = np.where(
        (fraud_bool == 1) & mobile_os,
        rng.uniform(0, 2, n),
        np.where(mobile_os, rng.normal(14, 4, n).clip(5), np.nan),
    )
    password_pasted = np.where(fraud_bool == 1, rng.binomial(1, 0.82, n), rng.binomial(1, 0.05, n))
    typing_wpm = np.where(password_pasted == 1, rng.uniform(180, 320, n), rng.normal(42, 12, n).clip(20, 90))
    dwell_time = np.where(password_pasted == 1, rng.uniform(10, 30, n), rng.normal(200, 50, n).clip(80, 350))
    failed_login = np.where(fraud_bool == 1, rng.poisson(3, n), rng.poisson(0.3, n))
    nav_velocity = np.where(fraud_bool == 1, rng.uniform(5, 12, n), rng.normal(2.0, 0.6, n).clip(1))
    session_len = np.where(fraud_bool == 1, rng.uniform(0.5, 5, n), rng.lognormal(2.5, 0.5, n))

    device_dist_emails = np.where(fraud_bool == 1, rng.poisson(18, n), rng.poisson(1.5, n))
    device_dist_accts = np.where(fraud_bool == 1, rng.poisson(5, n), rng.poisson(1, n))
    device_fraud_cnt = np.where(fraud_bool == 1, rng.poisson(4, n), 0)

    customer_age = rng.integers(22, 68, n)
    acct_age_months = rng.integers(1, 120, n)
    days_since_pw_reset = rng.lognormal(4, 0.8, n).astype(int)
    email_free = rng.binomial(1, 0.7, n)
    name_email_sim = np.where(fraud_bool == 1, rng.uniform(0.2, 0.6, n), rng.uniform(0.5, 0.95, n))
    phone_home_valid = rng.binomial(1, 0.9, n)
    phone_mobile_valid = rng.binomial(1, 0.95, n)
    failed_mfa = np.where(fraud_bool == 1, rng.binomial(2, 0.3, n), 0)
    mfa_changed = np.where(fraud_bool == 1, rng.binomial(1, 0.2, n), 0)
    profile_changed = np.where(fraud_bool == 1, rng.binomial(1, 0.65, n), rng.binomial(1, 0.02, n))
    time_to_change = np.where(
        profile_changed == 1,
        np.where(fraud_bool == 1, rng.uniform(10, 300, n), rng.uniform(600, 3600, n)),
        np.nan,
    )

    vel_6h = np.where(fraud_bool == 1, rng.poisson(6, n), rng.poisson(0.4, n))
    vel_24h = np.where(fraud_bool == 1, rng.poisson(15, n), rng.poisson(2, n))
    vel_4w = np.where(fraud_bool == 1, rng.poisson(40, n), rng.poisson(25, n))
    failed_tx_1h = np.where(fraud_bool == 1, rng.poisson(2, n), 0)
    tx_ratio = np.where(fraud_bool == 1, rng.uniform(5, 25, n), rng.uniform(0.8, 2.5, n))
    new_payee = np.where(fraud_bool == 1, rng.binomial(1, 0.72, n), rng.binomial(1, 0.05, n))
    payee_payout_time = np.where(
        new_payee == 1,
        np.where(fraud_bool == 1, rng.uniform(2, 30, n), rng.uniform(60, 1440, n)),
        np.nan,
    )
    promo_fails = np.where(fraud_bool == 1, rng.poisson(2, n), 0)
    saved_pm_count = np.where(fraud_bool == 1, rng.integers(0, 2, n), rng.integers(1, 4, n))
    payment_type = rng.choice(["credit_card", "debit_card", "net_banking", "upi"], n, p=[0.3, 0.3, 0.2, 0.2])
    month = rng.integers(1, 13, n)

    user_agent_parsed = [
        ua.split(" ")[2].strip("(") + "," + ua.split(" ")[3]
        if "Linux" in ua or "Android" in ua
        else "Windows,Chrome"
        for ua in ua_raw
    ]

    df = pd.DataFrame(
        {
            "ip_address_asn": rng.integers(1000, 99999, n),
            "is_known_vpn_or_proxy": is_known_vpn,
            "ip_distance_from_last_login_km": ip_distance.astype(int),
            "speed_of_travel_kmh": speed.astype(int),
            "is_new_device_uid": is_new_device,
            "device_fingerprint_canvas": canvas,
            "device_fingerprint_webgl": webgl,
            "user_agent_raw": ua_raw,
            "user_agent_parsed": user_agent_parsed,
            "network_connection_type": network_type,
            "device_os": device_os,
            "source": source,
            "is_foreign_ip": is_foreign_ip,
            "country_mismatch_with_profile": country_mismatch,
            "distinct_ip_zip_count_1w_4w": np.where(fraud_bool == 1, rng.poisson(7, n), rng.poisson(1, n)),
            "login_hour_of_day_local": login_hour,
            "login_hour_anomaly_score": anomaly_score.round(2),
            "mouse_movement_velocity": mouse_vel.astype(int),
            "touch_event_radius": touch_radius.round(1),
            "typing_speed_wpm": typing_wpm.astype(int),
            "keystroke_dwell_time": dwell_time.astype(int),
            "failed_login_attempts_session": failed_login,
            "password_pasted": password_pasted,
            "page_navigation_velocity": nav_velocity.round(1),
            "session_length_in_minutes": session_len.round(1),
            "keep_alive_session": rng.binomial(1, 0.6, n),
            "device_distinct_emails_8w": device_dist_emails,
            "device_distinct_accounts_attempted_24h": device_dist_accts,
            "device_fraud_count": device_fraud_cnt,
            "customer_age": customer_age,
            "account_age_months": acct_age_months,
            "days_since_last_login": days_since_last.round(2),
            "days_since_password_reset": days_since_pw_reset,
            "email_is_free": email_free,
            "name_email_similarity": name_email_sim.round(2),
            "phone_home_valid": phone_home_valid,
            "phone_mobile_valid": phone_mobile_valid,
            "failed_mfa_attempts_count": failed_mfa,
            "mfa_type_changed": mfa_changed,
            "profile_details_changed_flag": profile_changed,
            "time_to_profile_change_seconds": time_to_change.round(0),
            "velocity_6h": vel_6h,
            "velocity_24h": vel_24h,
            "velocity_4w": vel_4w,
            "failed_tx_count_1h": failed_tx_1h,
            "tx_amount_vs_historical_avg_ratio": tx_ratio.round(1),
            "new_payee_added": new_payee,
            "time_from_payee_addition_to_payout_minutes": payee_payout_time.round(0),
            "sequential_failed_promocode_attempts": promo_fails,
            "saved_payment_methods_count": saved_pm_count,
            "payment_type": payment_type,
            "month": month,
            "fraud_bool": fraud_bool,
        }
    )

    if noise_rate > 0:
        fraud_idx = df.index[df["fraud_bool"] == 1].to_numpy()
        real_idx = df.index[df["fraud_bool"] == 0].to_numpy()
        noisy_fraud = rng.choice(fraud_idx, size=int(len(fraud_idx) * noise_rate), replace=False)
        noisy_real = rng.choice(real_idx, size=int(len(real_idx) * noise_rate), replace=False)

        # Subtle fraud: fraudulent rows that look closer to a legitimate session.
        df.loc[noisy_fraud, "is_known_vpn_or_proxy"] = rng.binomial(1, 0.08, len(noisy_fraud))
        df.loc[noisy_fraud, "is_foreign_ip"] = rng.binomial(1, 0.05, len(noisy_fraud))
        df.loc[noisy_fraud, "country_mismatch_with_profile"] = rng.binomial(1, 0.05, len(noisy_fraud))
        df.loc[noisy_fraud, "ip_distance_from_last_login_km"] = rng.exponential(90, len(noisy_fraud)).astype(int)
        df.loc[noisy_fraud, "speed_of_travel_kmh"] = rng.exponential(15, len(noisy_fraud)).astype(int)
        df.loc[noisy_fraud, "is_new_device_uid"] = rng.binomial(1, 0.25, len(noisy_fraud))
        df.loc[noisy_fraud, "distinct_ip_zip_count_1w_4w"] = rng.poisson(1.5, len(noisy_fraud))
        df.loc[noisy_fraud, "login_hour_of_day_local"] = rng.integers(8, 23, len(noisy_fraud))
        df.loc[noisy_fraud, "login_hour_anomaly_score"] = rng.beta(0.7, 2.0, len(noisy_fraud)).round(2)
        df.loc[noisy_fraud, "mouse_movement_velocity"] = rng.normal(190, 90, len(noisy_fraud)).clip(0).astype(int)
        df.loc[noisy_fraud, "typing_speed_wpm"] = rng.normal(52, 22, len(noisy_fraud)).clip(20, 140).astype(int)
        df.loc[noisy_fraud, "keystroke_dwell_time"] = rng.normal(170, 70, len(noisy_fraud)).clip(40, 380).astype(int)
        df.loc[noisy_fraud, "password_pasted"] = rng.binomial(1, 0.18, len(noisy_fraud))
        df.loc[noisy_fraud, "failed_login_attempts_session"] = rng.poisson(0.8, len(noisy_fraud))
        df.loc[noisy_fraud, "page_navigation_velocity"] = rng.normal(2.4, 1.0, len(noisy_fraud)).clip(0.5).round(1)
        df.loc[noisy_fraud, "session_length_in_minutes"] = rng.lognormal(2.2, 0.7, len(noisy_fraud)).round(1)
        df.loc[noisy_fraud, "device_distinct_emails_8w"] = rng.poisson(2.0, len(noisy_fraud))
        df.loc[noisy_fraud, "device_distinct_accounts_attempted_24h"] = rng.poisson(1.2, len(noisy_fraud))
        df.loc[noisy_fraud, "device_fraud_count"] = rng.binomial(1, 0.08, len(noisy_fraud))
        df.loc[noisy_fraud, "name_email_similarity"] = rng.uniform(0.45, 0.9, len(noisy_fraud)).round(2)
        df.loc[noisy_fraud, "failed_mfa_attempts_count"] = rng.binomial(1, 0.08, len(noisy_fraud))
        df.loc[noisy_fraud, "mfa_type_changed"] = rng.binomial(1, 0.04, len(noisy_fraud))
        df.loc[noisy_fraud, "profile_details_changed_flag"] = rng.binomial(1, 0.12, len(noisy_fraud))
        df.loc[noisy_fraud, "velocity_6h"] = rng.poisson(1.0, len(noisy_fraud))
        df.loc[noisy_fraud, "velocity_24h"] = rng.poisson(3.0, len(noisy_fraud))
        df.loc[noisy_fraud, "failed_tx_count_1h"] = rng.binomial(1, 0.08, len(noisy_fraud))
        df.loc[noisy_fraud, "tx_amount_vs_historical_avg_ratio"] = rng.uniform(1.0, 4.0, len(noisy_fraud)).round(1)
        df.loc[noisy_fraud, "new_payee_added"] = rng.binomial(1, 0.18, len(noisy_fraud))
        df.loc[noisy_fraud, "sequential_failed_promocode_attempts"] = rng.poisson(0.2, len(noisy_fraud))

        # Hard negatives: legitimate rows with suspicious-looking signals.
        df.loc[noisy_real, "is_known_vpn_or_proxy"] = rng.binomial(1, 0.35, len(noisy_real))
        df.loc[noisy_real, "is_foreign_ip"] = rng.binomial(1, 0.22, len(noisy_real))
        df.loc[noisy_real, "country_mismatch_with_profile"] = rng.binomial(1, 0.18, len(noisy_real))
        df.loc[noisy_real, "ip_distance_from_last_login_km"] = rng.uniform(300, 6000, len(noisy_real)).astype(int)
        df.loc[noisy_real, "speed_of_travel_kmh"] = rng.uniform(40, 650, len(noisy_real)).astype(int)
        df.loc[noisy_real, "is_new_device_uid"] = rng.binomial(1, 0.55, len(noisy_real))
        df.loc[noisy_real, "distinct_ip_zip_count_1w_4w"] = rng.poisson(3.5, len(noisy_real))
        df.loc[noisy_real, "login_hour_of_day_local"] = rng.choice([0, 1, 2, 3, 4, 5, 23], len(noisy_real))
        df.loc[noisy_real, "login_hour_anomaly_score"] = rng.beta(1.5, 1.0, len(noisy_real)).round(2)
        df.loc[noisy_real, "mouse_movement_velocity"] = rng.choice([0, 5, 10, 25], len(noisy_real), p=[0.25, 0.25, 0.25, 0.25])
        df.loc[noisy_real, "typing_speed_wpm"] = rng.uniform(110, 250, len(noisy_real)).astype(int)
        df.loc[noisy_real, "keystroke_dwell_time"] = rng.uniform(25, 120, len(noisy_real)).astype(int)
        df.loc[noisy_real, "password_pasted"] = rng.binomial(1, 0.35, len(noisy_real))
        df.loc[noisy_real, "failed_login_attempts_session"] = rng.poisson(1.5, len(noisy_real))
        df.loc[noisy_real, "page_navigation_velocity"] = rng.uniform(3.0, 8.0, len(noisy_real)).round(1)
        df.loc[noisy_real, "session_length_in_minutes"] = rng.uniform(1.0, 8.0, len(noisy_real)).round(1)
        df.loc[noisy_real, "device_distinct_emails_8w"] = rng.poisson(6.0, len(noisy_real))
        df.loc[noisy_real, "device_distinct_accounts_attempted_24h"] = rng.poisson(2.2, len(noisy_real))
        df.loc[noisy_real, "device_fraud_count"] = rng.binomial(2, 0.15, len(noisy_real))
        df.loc[noisy_real, "name_email_similarity"] = rng.uniform(0.25, 0.75, len(noisy_real)).round(2)
        df.loc[noisy_real, "failed_mfa_attempts_count"] = rng.binomial(2, 0.12, len(noisy_real))
        df.loc[noisy_real, "mfa_type_changed"] = rng.binomial(1, 0.08, len(noisy_real))
        df.loc[noisy_real, "profile_details_changed_flag"] = rng.binomial(1, 0.18, len(noisy_real))
        df.loc[noisy_real, "velocity_6h"] = rng.poisson(3.0, len(noisy_real))
        df.loc[noisy_real, "velocity_24h"] = rng.poisson(8.0, len(noisy_real))
        df.loc[noisy_real, "failed_tx_count_1h"] = rng.binomial(2, 0.12, len(noisy_real))
        df.loc[noisy_real, "tx_amount_vs_historical_avg_ratio"] = rng.uniform(2.0, 12.0, len(noisy_real)).round(1)
        df.loc[noisy_real, "new_payee_added"] = rng.binomial(1, 0.30, len(noisy_real))
        df.loc[noisy_real, "sequential_failed_promocode_attempts"] = rng.poisson(0.8, len(noisy_real))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(["fraud_bool"])
        missing_mask = rng.random((len(df), len(numeric_cols))) < 0.01
        df[numeric_cols] = df[numeric_cols].astype(float)
        df[numeric_cols] = df[numeric_cols].mask(missing_mask)

    if label_noise_rate > 0:
        flip_count = int(len(df) * label_noise_rate)
        flip_idx = rng.choice(df.index.to_numpy(), size=flip_count, replace=False)
        df.loc[flip_idx, "fraud_bool"] = 1 - df.loc[flip_idx, "fraud_bool"]

    output = Path(output_path) if output_path is not None else RAW_DATA_DIR / "ato_behavioral_dataset.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df
