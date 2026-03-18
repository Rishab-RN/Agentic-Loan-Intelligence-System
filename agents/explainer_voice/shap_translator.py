"""
ALIS — ExplainerVoice: SHAP-to-Human Translator
==================================================
Maps machine-learned SHAP feature names to warm, plain-language
explanations in Kannada, Hindi, and English.

These translations are hand-crafted, not machine-translated — because
machine translation of financial concepts in Kannada produces gibberish
that no auto driver in Ballari would understand. We validated these
with native speakers.

The tone is: trusted elder brother who studied finance and genuinely
wants to help. Never formal. Never condescending.
"""

from typing import Optional

# ─── Feature Translation Dictionary ─────────────────────────────────────────
# Each feature maps to:
#   name: human-readable feature name
#   positive: what it means when this feature HELPS the score
#   negative: what it means when this feature HURTS the score
#   advice: actionable improvement advice

TRANSLATIONS = {
    "utility_bill_payment_consistency": {
        "english": {
            "name": "Electricity/water bill payment history",
            "positive": "You've been paying your electricity and water bills regularly — this is your strongest point",
            "negative": "Your electricity and water bills haven't been paid on time consistently — this is the biggest thing holding your score back",
            "advice": "Pay your electricity bill on time every month for the next {months} months. Even if the bill is small, on-time payment matters more than the amount",
        },
        "kannada": {
            "name": "ವಿದ್ಯುತ್/ನೀರಿನ ಬಿಲ್ ಪಾವತಿ ಇತಿಹಾಸ",
            "positive": "ನೀವು ವಿದ್ಯುತ್ ಮತ್ತು ನೀರಿನ ಬಿಲ್ ಅನ್ನು ನಿಯಮಿತವಾಗಿ ಪಾವತಿಸುತ್ತಿದ್ದೀರಿ — ಇದು ನಿಮ್ಮ ಅತ್ಯಂತ ಬಲವಾದ ಅಂಶ",
            "negative": "ನಿಮ್ಮ ವಿದ್ಯುತ್ ಮತ್ತು ನೀರಿನ ಬಿಲ್ ಸಮಯಕ್ಕೆ ಪಾವತಿಯಾಗಿಲ್ಲ — ಇದು ನಿಮ್ಮ ಸ್ಕೋರ್ ಕಡಿಮೆ ಮಾಡುತ್ತಿರುವ ಪ್ರಮುಖ ವಿಷಯ",
            "advice": "ಮುಂದಿನ {months} ತಿಂಗಳು ವಿದ್ಯುತ್ ಬಿಲ್ ಅನ್ನು ಸಮಯಕ್ಕೆ ಪಾವತಿಸಿ. ಬಿಲ್ ಮೊತ್ತ ಚಿಕ್ಕದಾದರೂ ಪರವಾಗಿಲ್ಲ, ಸಮಯಕ್ಕೆ ಕಟ್ಟುವುದು ಮುಖ್ಯ",
        },
        "hindi": {
            "name": "बिजली/पानी बिल भुगतान का इतिहास",
            "positive": "आप अपने बिजली और पानी के बिल नियमित रूप से भर रहे हैं — यह आपका सबसे मजबूत पक्ष है",
            "negative": "आपके बिजली और पानी के बिल समय पर नहीं भरे गए हैं — यही चीज़ आपका स्कोर सबसे ज़्यादा कम कर रही है",
            "advice": "अगले {months} महीने बिजली का बिल समय पर भरें। बिल छोटा हो तो भी कोई बात नहीं, समय पर भरना ज़्यादा ज़रूरी है",
        },
    },
    "upi_txn_frequency_30d": {
        "english": {
            "name": "How actively you use UPI (Google Pay, PhonePe, etc.)",
            "positive": "You use UPI regularly for payments — this shows you're financially active",
            "negative": "Your UPI usage is low — more digital transactions help build your financial profile",
            "advice": "Use UPI for your daily purchases (tea, vegetables, fuel) — even ₹10-20 transactions count. Try to make at least {target} transactions per day",
        },
        "kannada": {
            "name": "ನೀವು UPI (Google Pay, PhonePe) ಎಷ್ಟು ಬಳಸುತ್ತೀರಿ",
            "positive": "ನೀವು UPI ನಿಯಮಿತವಾಗಿ ಬಳಸುತ್ತಿದ್ದೀರಿ — ಇದು ನಿಮ್ಮ ಆರ್ಥಿಕ ಸಕ್ರಿಯತೆ ತೋರಿಸುತ್ತದೆ",
            "negative": "ನಿಮ್ಮ UPI ಬಳಕೆ ಕಡಿಮೆ ಇದೆ — ಹೆಚ್ಚು ಡಿಜಿಟಲ್ ವ್ಯವಹಾರ ಮಾಡಿದರೆ ನಿಮ್ಮ ಪ್ರೊಫೈಲ್ ಬಲಗೊಳ್ಳುತ್ತದೆ",
            "advice": "ದಿನನಿತ್ಯದ ಖರೀದಿಗೆ UPI ಬಳಸಿ (ಚಹಾ, ತರಕಾರಿ, ಪೆಟ್ರೋಲ್) — ₹10-20 ವ್ಯವಹಾರವೂ ಲೆಕ್ಕಕ್ಕೆ ಬರುತ್ತದೆ. ದಿನಕ್ಕೆ ಕನಿಷ್ಠ {target} ವ್ಯವಹಾರ ಮಾಡಲು ಪ್ರಯತ್ನಿಸಿ",
        },
        "hindi": {
            "name": "आप UPI (Google Pay, PhonePe) कितना इस्तेमाल करते हैं",
            "positive": "आप नियमित रूप से UPI से भुगतान करते हैं — यह दिखाता है कि आप आर्थिक रूप से सक्रिय हैं",
            "negative": "आपका UPI उपयोग कम है — ज़्यादा डिजिटल लेनदेन आपकी प्रोफ़ाइल मजबूत करते हैं",
            "advice": "रोज़मर्रा की खरीदारी के लिए UPI इस्तेमाल करें (चाय, सब्ज़ी, पेट्रोल) — ₹10-20 का लेनदेन भी मायने रखता है। रोज़ कम से कम {target} लेनदेन करने की कोशिश करें",
        },
    },
    "savings_behavior_score": {
        "english": {
            "name": "How regularly you save money",
            "positive": "You regularly transfer money to savings — this shows financial discipline",
            "negative": "You don't save regularly — even small, consistent savings make a big difference",
            "advice": "Transfer even ₹200-500 to your savings account every month. Set a reminder on the 1st of each month. Consistency matters more than amount",
        },
        "kannada": {
            "name": "ನೀವು ಎಷ್ಟು ನಿಯಮಿತವಾಗಿ ಹಣ ಉಳಿಸುತ್ತೀರಿ",
            "positive": "ನೀವು ನಿಯಮಿತವಾಗಿ ಉಳಿತಾಯ ಖಾತೆಗೆ ಹಣ ವರ್ಗಾಯಿಸುತ್ತೀರಿ — ಇದು ಆರ್ಥಿಕ ಶಿಸ್ತು ತೋರಿಸುತ್ತದೆ",
            "negative": "ನೀವು ನಿಯಮಿತವಾಗಿ ಉಳಿಸುತ್ತಿಲ್ಲ — ಸಣ್ಣ, ನಿರಂತರ ಉಳಿತಾಯವೂ ದೊಡ್ಡ ವ್ಯತ್ಯಾಸ ಮಾಡುತ್ತದೆ",
            "advice": "ಪ್ರತಿ ತಿಂಗಳು ₹200-500 ಉಳಿತಾಯ ಖಾತೆಗೆ ವರ್ಗಾಯಿಸಿ. ಪ್ರತಿ ತಿಂಗಳ 1 ನೇ ತಾರೀಖಿಗೆ ರಿಮೈಂಡರ್ ಇಡಿ. ಮೊತ್ತಕ್ಕಿಂತ ನಿಯಮಿತತೆ ಮುಖ್ಯ",
        },
        "hindi": {
            "name": "आप कितनी नियमितता से बचत करते हैं",
            "positive": "आप नियमित रूप से बचत खाते में पैसा डालते हैं — यह आर्थिक अनुशासन दिखाता है",
            "negative": "आप नियमित बचत नहीं कर रहे — छोटी, लगातार बचत भी बड़ा फ़र्क डालती है",
            "advice": "हर महीने ₹200-500 बचत खाते में डालें। हर महीने की 1 तारीख को रिमाइंडर लगा लें। रकम से ज़्यादा नियमितता ज़रूरी है",
        },
    },
    "income_estimate_monthly": {
        "english": {
            "name": "Your estimated monthly income",
            "positive": "Your monthly income level supports this loan comfortably",
            "negative": "Your monthly income is on the lower side for this loan size — a smaller loan may be more comfortable",
            "advice": "Consider applying for a smaller loan amount (₹{suggested_amount}) that fits within your monthly income",
        },
        "kannada": {
            "name": "ನಿಮ್ಮ ಅಂದಾಜು ಮಾಸಿಕ ಆದಾಯ",
            "positive": "ನಿಮ್ಮ ತಿಂಗಳ ಆದಾಯ ಈ ಸಾಲಕ್ಕೆ ಸಾಕಾಗುತ್ತದೆ",
            "negative": "ಈ ಸಾಲದ ಮೊತ್ತಕ್ಕೆ ನಿಮ್ಮ ಆದಾಯ ಸ್ವಲ್ಪ ಕಡಿಮೆ — ಚಿಕ್ಕ ಸಾಲ ಹೆಚ್ಚು ಸೂಕ್ತ",
            "advice": "ನಿಮ್ಮ ಮಾಸಿಕ ಆದಾಯಕ್ಕೆ ಹೊಂದುವ ₹{suggested_amount} ಸಾಲಕ್ಕೆ ಅರ್ಜಿ ಸಲ್ಲಿಸಿ",
        },
        "hindi": {
            "name": "आपकी अनुमानित मासिक आय",
            "positive": "आपकी मासिक आय इस लोन के लिए पर्याप्त है",
            "negative": "इस लोन की रकम के लिए आपकी आय थोड़ी कम है — छोटा लोन ज़्यादा आरामदायक होगा",
            "advice": "अपनी मासिक आय के हिसाब से ₹{suggested_amount} का लोन लें",
        },
    },
    "income_volatility_cv": {
        "english": {
            "name": "How steady your income is month-to-month",
            "positive": "Your income is fairly steady each month — lenders like predictability",
            "negative": "Your income varies a lot from month to month — this makes lenders nervous",
            "advice": "Try to diversify your income. Even a small side job (like evening tuitions or weekend deliveries) that brings in ₹2000-3000 monthly helps smooth out the ups and downs",
        },
        "kannada": {
            "name": "ನಿಮ್ಮ ಆದಾಯ ಎಷ್ಟು ಸ್ಥಿರವಾಗಿದೆ",
            "positive": "ನಿಮ್ಮ ಆದಾಯ ಪ್ರತಿ ತಿಂಗಳು ಸಮಂಜಸವಾಗಿ ಸ್ಥಿರವಾಗಿದೆ — ಸಾಲದಾತರಿಗೆ ಇದು ಇಷ್ಟ",
            "negative": "ನಿಮ್ಮ ಆದಾಯ ತಿಂಗಳಿಂದ ತಿಂಗಳಿಗೆ ತುಂಬಾ ಬದಲಾಗುತ್ತದೆ — ಇದು ಸಾಲ ಮಂಜೂರಾತಿಗೆ ಕಷ್ಟ",
            "advice": "ಆದಾಯವನ್ನು ವೈವಿಧ್ಯಗೊಳಿಸಿ. ಸಂಜೆ ಟ್ಯೂಷನ್ ಅಥವಾ ವೀಕೆಂಡ್ ಡೆಲಿವರಿ ಮಾಡಿ ₹2000-3000 ಹೆಚ್ಚುವರಿ ಗಳಿಸಿ — ಇದು ಆದಾಯ ಸ್ಥಿರಗೊಳಿಸುತ್ತದೆ",
        },
        "hindi": {
            "name": "आपकी आय हर महीने कितनी स्थिर है",
            "positive": "आपकी आय हर महीने काफ़ी स्थिर रहती है — इसे बैंक पसंद करते हैं",
            "negative": "आपकी आय हर महीने बहुत बदलती है — इससे बैंक को चिंता होती है",
            "advice": "आय को स्थिर करने की कोशिश करें। शाम को ट्यूशन या वीकेंड डिलीवरी से ₹2000-3000 कमाएं — यह उतार-चढ़ाव कम करता है",
        },
    },
    "bnpl_outstanding_ratio": {
        "english": {
            "name": "Your existing EMI/Buy-Now-Pay-Later burden",
            "positive": "You don't have too much borrowed money outstanding — your debt is manageable",
            "negative": "You have too many EMIs or Buy-Now-Pay-Later balances compared to your income",
            "advice": "Pay off your existing EMIs first, especially the small ones. Reducing your outstanding balance by ₹{reduce_by} will significantly improve your score",
        },
        "kannada": {
            "name": "ನಿಮ್ಮ ಈಗಿರುವ EMI/ಸಾಲದ ಹೊರೆ",
            "positive": "ನಿಮ್ಮ ಮೇಲೆ ಹೆಚ್ಚು ಸಾಲದ ಹೊರೆ ಇಲ್ಲ — ಇದು ಒಳ್ಳೆಯ ಸಂಕೇತ",
            "negative": "ನಿಮ್ಮ ಆದಾಯಕ್ಕೆ ಹೋಲಿಸಿದರೆ ನಿಮ್ಮ EMI ಮತ್ತು ಸಾಲ ಹೊರೆ ಹೆಚ್ಚು",
            "advice": "ಮೊದಲು ಈಗಿರುವ ಚಿಕ್ಕ EMI ಗಳನ್ನು ಮುಗಿಸಿ. ₹{reduce_by} ಬಾಕಿ ಕಡಿಮೆ ಮಾಡಿದರೆ ನಿಮ್ಮ ಸ್ಕೋರ್ ಗಮನಾರ್ಹವಾಗಿ ಸುಧಾರಿಸುತ್ತದೆ",
        },
        "hindi": {
            "name": "आपकी मौजूदा EMI/उधार का बोझ",
            "positive": "आप पर ज़्यादा उधार का बोझ नहीं है — यह अच्छा संकेत है",
            "negative": "आपकी आय के मुकाबले आपकी EMI और उधार बहुत ज़्यादा हैं",
            "advice": "पहले छोटी EMI ख़त्म करें। ₹{reduce_by} का बकाया कम करने से आपका स्कोर काफ़ी सुधरेगा",
        },
    },
    "multi_loan_app_count": {
        "english": {
            "name": "Number of loan apps on your phone",
            "positive": "You don't have too many loan apps — this shows you're not desperately seeking credit",
            "negative": "You have several loan apps installed — banks see this as a sign of financial stress",
            "advice": "Uninstall loan apps you don't use. Keep only 1-2 trusted ones. Having {current} apps looks like you're desperately seeking loans",
        },
        "kannada": {
            "name": "ನಿಮ್ಮ ಫೋನ್‌ನಲ್ಲಿ ಇರುವ ಸಾಲ ಆಪ್‌ಗಳ ಸಂಖ್ಯೆ",
            "positive": "ನಿಮ್ಮಲ್ಲಿ ಹೆಚ್ಚು ಸಾಲ ಆಪ್‌ಗಳಿಲ್ಲ — ಇದು ಒಳ್ಳೆಯ ಸಂಕೇತ",
            "negative": "ನಿಮ್ಮ ಫೋನ್‌ನಲ್ಲಿ ಹಲವು ಸಾಲ ಆಪ್‌ಗಳಿವೆ — ಬ್ಯಾಂಕ್‌ಗಳು ಇದನ್ನು ಆರ್ಥಿಕ ಒತ್ತಡದ ಸಂಕೇತವಾಗಿ ನೋಡುತ್ತವೆ",
            "advice": "ಬಳಸದ ಸಾಲ ಆಪ್‌ಗಳನ್ನು ಡಿಲೀಟ್ ಮಾಡಿ. 1-2 ನಂಬಿಕೆಯ ಆಪ್‌ಗಳನ್ನು ಮಾತ್ರ ಇಟ್ಟುಕೊಳ್ಳಿ. {current} ಆಪ್‌ಗಳಿರುವುದು ನಿಮ್ಮನ್ನು ಹತಾಶವಾಗಿ ತೋರಿಸುತ್ತದೆ",
        },
        "hindi": {
            "name": "आपके फ़ोन में कितने लोन ऐप हैं",
            "positive": "आपके पास ज़्यादा लोन ऐप नहीं हैं — यह अच्छा संकेत है",
            "negative": "आपके फ़ोन में कई लोन ऐप हैं — बैंक इसे आर्थिक तनाव का संकेत मानते हैं",
            "advice": "जो लोन ऐप इस्तेमाल नहीं करते, उन्हें हटा दें। सिर्फ़ 1-2 भरोसेमंद ऐप रखें। {current} ऐप होना अच्छा नहीं दिखता",
        },
    },
    "device_tenure_months": {
        "english": {
            "name": "How long you've been using your current phone",
            "positive": "You've been using your phone for a while — this stability is a positive signal",
            "negative": "You've changed your phone recently — longer phone usage shows stability",
            "advice": "Keep using your current phone. After {months_to_go} more months, this factor will turn positive",
        },
        "kannada": {
            "name": "ನೀವು ಎಷ್ಟು ತಿಂಗಳಿಂದ ಈ ಫೋನ್ ಬಳಸುತ್ತಿದ್ದೀರಿ",
            "positive": "ನೀವು ಈ ಫೋನ್ ಅನ್ನು ಬಹಳ ಕಾಲದಿಂದ ಬಳಸುತ್ತಿದ್ದೀರಿ — ಈ ಸ್ಥಿರತೆ ಒಳ್ಳೆಯ ಸಂಕೇತ",
            "negative": "ನೀವು ಇತ್ತೀಚೆಗೆ ಫೋನ್ ಬದಲಿಸಿದ್ದೀರಿ — ದೀರ್ಘ ಫೋನ್ ಬಳಕೆ ಸ್ಥಿರತೆ ತೋರಿಸುತ್ತದೆ",
            "advice": "ಈ ಫೋನ್ ಅನ್ನು ಬಳಸುತ್ತಿರಿ. {months_to_go} ತಿಂಗಳ ನಂತರ ಈ ಅಂಶ ನಿಮ್ಮ ಪರವಾಗಿ ಬರುತ್ತದೆ",
        },
        "hindi": {
            "name": "आप कितने समय से यह फ़ोन इस्तेमाल कर रहे हैं",
            "positive": "आप लंबे समय से यह फ़ोन इस्तेमाल कर रहे हैं — यह स्थिरता का अच्छा संकेत है",
            "negative": "आपने हाल ही में फ़ोन बदला है — लंबे समय तक एक फ़ोन इस्तेमाल करना अच्छा दिखता है",
            "advice": "यही फ़ोन इस्तेमाल करते रहें। {months_to_go} और महीने बाद यह पॉइंट आपके पक्ष में आ जाएगा",
        },
    },
    "mobile_recharge_regularity": {
        "english": {
            "name": "How regularly you recharge your phone",
            "positive": "You recharge your phone at regular intervals — this shows consistency",
            "negative": "Your phone recharges are irregular — regular recharges show stable phone usage",
            "advice": "Recharge your phone on a fixed date each month (like the 1st or 15th). Regular recharges build your profile",
        },
        "kannada": {
            "name": "ನೀವು ಎಷ್ಟು ನಿಯಮಿತವಾಗಿ ಫೋನ್ ರೀಚಾರ್ಜ್ ಮಾಡುತ್ತೀರಿ",
            "positive": "ನೀವು ನಿಯಮಿತ ಅಂತರದಲ್ಲಿ ಫೋನ್ ರೀಚಾರ್ಜ್ ಮಾಡುತ್ತೀರಿ — ಇದು ಸ್ಥಿರತೆ ತೋರಿಸುತ್ತದೆ",
            "negative": "ನಿಮ್ಮ ಫೋನ್ ರೀಚಾರ್ಜ್ ಅನಿಯಮಿತ — ನಿಯಮಿತ ರೀಚಾರ್ಜ್ ನಿಮ್ಮ ಪ್ರೊಫೈಲ್ ಬಲಗೊಳಿಸುತ್ತದೆ",
            "advice": "ಪ್ರತಿ ತಿಂಗಳು ಒಂದು ನಿರ್ದಿಷ್ಟ ದಿನ (1 ಅಥವಾ 15 ನೇ ತಾರೀಖು) ರೀಚಾರ್ಜ್ ಮಾಡಿ",
        },
        "hindi": {
            "name": "आप कितनी नियमितता से फ़ोन रिचार्ज करते हैं",
            "positive": "आप नियमित अंतराल पर फ़ोन रिचार्ज करते हैं — यह ज़िम्मेदारी दिखाता है",
            "negative": "आपका फ़ोन रिचार्ज अनियमित है — नियमित रिचार्ज प्रोफ़ाइल मजबूत करता है",
            "advice": "हर महीने एक तय तारीख (जैसे 1 या 15) को रिचार्ज करें। नियमित रिचार्ज से प्रोफ़ाइल अच्छी बनती है",
        },
    },
    "upi_merchant_diversity_score": {
        "english": {
            "name": "Variety of shops/merchants you pay through UPI",
            "positive": "You pay at many different shops via UPI — this shows genuine, diverse spending",
            "negative": "You only transact with very few merchants — more variety shows genuine commerce",
            "advice": "Use UPI at different shops — your regular kirana store, fuel station, vegetable vendor. Variety matters",
        },
        "kannada": {
            "name": "ನೀವು UPI ಮೂಲಕ ಎಷ್ಟು ವಿವಿಧ ಅಂಗಡಿಗಳಲ್ಲಿ ಪಾವತಿಸುತ್ತೀರಿ",
            "positive": "ನೀವು ಅನೇಕ ಬೇರೆ ಬೇರೆ ಅಂಗಡಿಗಳಲ್ಲಿ UPI ಬಳಸುತ್ತೀರಿ — ಇದು ನಿಜವಾದ ವ್ಯಾಪಾರ ತೋರಿಸುತ್ತದೆ",
            "negative": "ನೀವು ಬಹಳ ಕಡಿಮೆ ಅಂಗಡಿಗಳಲ್ಲಿ ಮಾತ್ರ UPI ಬಳಸುತ್ತೀರಿ — ಹೆಚ್ಚು ವೈವಿಧ್ಯ ಬೇಕು",
            "advice": "ಬೇರೆ ಬೇರೆ ಅಂಗಡಿಗಳಲ್ಲಿ UPI ಬಳಸಿ — ಕಿರಾಣಿ, ಪೆಟ್ರೋಲ್ ಬಂಕ್, ತರಕಾರಿ ವ್ಯಾಪಾರಿ. ವೈವಿಧ್ಯ ಮುಖ್ಯ",
        },
        "hindi": {
            "name": "आप UPI से कितनी अलग-अलग दुकानों पर भुगतान करते हैं",
            "positive": "आप कई अलग-अलग दुकानों पर UPI इस्तेमाल करते हैं — यह असली खर्च दिखाता है",
            "negative": "आप बहुत कम दुकानों पर UPI इस्तेमाल करते हैं — ज़्यादा विविधता ज़रूरी है",
            "advice": "अलग-अलग दुकानों पर UPI इस्तेमाल करें — किराना, पेट्रोल, सब्ज़ी वाला। विविधता मायने रखती है",
        },
    },
    "evening_txn_ratio": {
        "english": {
            "name": "How much you transact late at night",
            "positive": "Your transaction timing is normal — most activity during business hours",
            "negative": "A lot of your transactions happen late at night — this is an unusual pattern",
            "advice": "Try to do more of your transactions during daytime hours (8 AM to 8 PM). Late-night transaction patterns can flag risk",
        },
        "kannada": {
            "name": "ರಾತ್ರಿ ತಡವಾಗಿ ಎಷ್ಟು ವ್ಯವಹಾರ ಮಾಡುತ್ತೀರಿ",
            "positive": "ನಿಮ್ಮ ವ್ಯವಹಾರ ಸಮಯ ಸಾಮಾನ್ಯವಾಗಿದೆ — ಹೆಚ್ಚಿನ ಚಟುವಟಿಕೆ ಹಗಲು ಸಮಯದಲ್ಲಿ",
            "negative": "ನಿಮ್ಮ ಹೆಚ್ಚಿನ ವ್ಯವಹಾರ ರಾತ್ರಿ ತಡವಾಗಿ ಆಗುತ್ತದೆ — ಇದು ಅಸಾಮಾನ್ಯ ಮಾದರಿ",
            "advice": "ಹೆಚ್ಚಿನ ವ್ಯವಹಾರ ಹಗಲು (ಬೆಳಿಗ್ಗೆ 8 ರಿಂದ ರಾತ್ರಿ 8) ಮಾಡಿ. ರಾತ್ರಿ ವ್ಯವಹಾರ ಅಪಾಯ ಸಂಕೇತವಾಗಬಹುದು",
        },
        "hindi": {
            "name": "रात को देर से कितना लेनदेन करते हैं",
            "positive": "आपका लेनदेन समय सामान्य है — ज़्यादातर काम दिन में होता है",
            "negative": "आपका बहुत सा लेनदेन देर रात को होता है — यह असामान्य पैटर्न है",
            "advice": "ज़्यादातर लेनदेन दिन में (सुबह 8 से रात 8) करें। देर रात का पैटर्न जोखिम का संकेत हो सकता है",
        },
    },
    "peer_transfer_reciprocity": {
        "english": {
            "name": "Balance between money you send and receive from friends/family",
            "positive": "You have a healthy balance of sending and receiving money — good social trust",
            "negative": "You're sending much more than you receive (or vice versa) — balanced relationships look better",
            "advice": "This naturally improves over time. Just continue normal financial relationships with friends and family",
        },
        "kannada": {
            "name": "ಸ್ನೇಹಿತರು/ಕುಟುಂಬದೊಂದಿಗೆ ಹಣ ಕಳಿಸುವ ಮತ್ತು ಪಡೆಯುವ ಸಮತೋಲನ",
            "positive": "ನೀವು ಹಣ ಕಳಿಸುವುದು ಮತ್ತು ಪಡೆಯುವುದು ಸಮತೋಲನದಲ್ಲಿದೆ — ಒಳ್ಳೆಯ ಸಾಮಾಜಿಕ ನಂಬಿಕೆ",
            "negative": "ನೀವು ಕಳಿಸುವ ಮತ್ತು ಪಡೆಯುವ ಹಣದ ಅನುಪಾತ ಅಸಮತೋಲನದಲ್ಲಿದೆ",
            "advice": "ಇದು ಕಾಲಕ್ರಮೇಣ ಸ್ವಾಭಾವಿಕವಾಗಿ ಸುಧಾರಿಸುತ್ತದೆ. ಸಾಮಾನ್ಯ ಆರ್ಥಿಕ ಸಂಬಂಧಗಳನ್ನು ಮುಂದುವರಿಸಿ",
        },
        "hindi": {
            "name": "दोस्तों/परिवार को पैसे भेजने और पाने का संतुलन",
            "positive": "आपका पैसा भेजना और पाना संतुलित है — अच्छा सामाजिक विश्वास",
            "negative": "आप बहुत ज़्यादा भेज रहे हैं या बहुत ज़्यादा पा रहे हैं — संतुलन बेहतर दिखता है",
            "advice": "यह समय के साथ खुद सुधरता है। बस सामान्य आर्थिक संबंध बनाए रखें",
        },
    },
}

SUPPORTED_LANGUAGES = {"english", "kannada", "hindi"}


# ─── Public API ──────────────────────────────────────────────────────────────

def translate_shap_to_language(
    shap_values: dict,
    language: str = "english",
) -> list[dict]:
    """
    Translate SHAP feature attributions to human-readable explanations.

    Parameters
    ----------
    shap_values : dict
        {feature_name: shap_value} from RiskMind's explainer.
    language : str
        "english", "kannada", or "hindi".

    Returns
    -------
    list of dicts, sorted by |shap_value| descending:
        feature: str
        name: str (human-readable in the chosen language)
        shap_value: float
        direction: "positive" | "negative"
        explanation: str
        advice: str (if negative direction)
    """
    if language not in SUPPORTED_LANGUAGES:
        language = "english"

    results = []
    for feature, shap_val in sorted(
        shap_values.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        if feature not in TRANSLATIONS:
            continue

        trans = TRANSLATIONS[feature].get(language, TRANSLATIONS[feature]["english"])
        direction = "positive" if shap_val > 0 else "negative"

        entry = {
            "feature": feature,
            "name": trans["name"],
            "shap_value": round(shap_val, 4),
            "direction": direction,
            "explanation": trans[direction],
        }

        if direction == "negative":
            entry["advice"] = trans.get("advice", "")

        results.append(entry)

    return results


def get_feature_name(feature: str, language: str = "english") -> str:
    """Get human-readable feature name in the specified language."""
    if feature in TRANSLATIONS:
        trans = TRANSLATIONS[feature].get(language, TRANSLATIONS[feature]["english"])
        return trans["name"]
    return feature
