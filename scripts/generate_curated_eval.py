"""Generate a diversified curated 100-query multilingual evaluation set."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "eval_queries.json"


LANGUAGE_ORDER = ["gu", "hi", "ta", "mr", "bn", "te", "kn", "ml", "pa", "en"]

TEMPLATES = {
    "gu": ["{topic} નો કેસ"],
    "hi": ["{topic} का मामला"],
    "ta": ["{topic} வழக்கு"],
    "mr": ["{topic} चा मामला"],
    "bn": ["{topic} মামলা"],
    "te": ["{topic} కేసు"],
    "kn": ["{topic} ಪ್ರಕರಣ"],
    "ml": ["{topic} കേസ്"],
    "pa": ["{topic} ਦਾ ਮਾਮਲਾ"],
    "en": ["{topic} case"],
}


INTENTS = [
    {
        "intent_id": "fraud_fake_receipts",
        "relevant_cases": [
            {"case_id": "36224681", "relevance": 3},
            {"case_id": "149802379", "relevance": 2},
            {"case_id": "108488019", "relevance": 1},
        ],
        "topics": {
            "gu": "નકલી રસીદો વડે પૈસા વસૂલવાની છેતરપિંડી",
            "hi": "नकली रसीदों से पैसे लेने वाली धोखाधड़ी",
            "ta": "போலி ரசீதுகள் மூலம் பணம் பெற்ற மோசடி",
            "mr": "खोट्या पावत्यांद्वारे पैसे उकळण्याची फसवणूक",
            "bn": "নকল রসিদ ব্যবহার করে টাকা নেওয়ার প্রতারণা",
            "te": "నకిలీ రసీదులతో డబ్బు వసూలు చేసిన మోసం",
            "kn": "ನಕಲಿ ರಸೀದಿಗಳ ಮೂಲಕ ಹಣ ವಸೂಲಿ ಮಾಡಿದ ಮೋಸ",
            "ml": "നകലി രസീതുകൾ ഉപയോഗിച്ച് പണം ഈടാക്കിയ തട്ടിപ്പ്",
            "pa": "ਨਕਲੀ ਰਸੀਦਾਂ ਨਾਲ ਪੈਸੇ ਲੈਣ ਵਾਲੀ ਧੋਖਾਧੜੀ",
            "en": "fraud involving fake receipts and collection of money",
        },
    },
    {
        "intent_id": "dowry_cruelty",
        "relevant_cases": [
            {"case_id": "30168989", "relevance": 3},
            {"case_id": "123961503", "relevance": 2},
            {"case_id": "105813417", "relevance": 1},
        ],
        "topics": {
            "gu": "દહેજની માંગ અને ઘરેલુ ક્રૂરતા",
            "hi": "दहेज की मांग और घरेलू क्रूरता",
            "ta": "வரதட்சணை கோரிக்கை மற்றும் குடும்பக் கொடுமை",
            "mr": "हुंड्याची मागणी आणि घरगुती क्रूरता",
            "bn": "পণের দাবি ও গার্হস্থ্য নির্যাতন",
            "te": "కట్నం డిమాండ్ మరియు గృహ క్రూరత్వం",
            "kn": "ಪಣದ ಬೇಡಿಕೆ ಮತ್ತು ಗೃಹ ಕ್ರೂರತೆ",
            "ml": "വരദക്ഷിണാവശ്യവും ഗാർഹിക ക്രൂരതയും",
            "pa": "ਦਹੇਜ ਦੀ ਮੰਗ ਅਤੇ ਘਰੇਲੂ ਕ੍ਰੂਰਤਾ",
            "en": "dowry demand and domestic cruelty",
        },
    },
    {
        "intent_id": "property_sale_deed_forgery",
        "relevant_cases": [
            {"case_id": "149802379", "relevance": 3},
            {"case_id": "172971596", "relevance": 2},
            {"case_id": "858350", "relevance": 1},
        ],
        "topics": {
            "gu": "જાળી વેચાણ દસ્તાવેજ અને મિલ્કત હસ્તાંતરણ છેતરપિંડી",
            "hi": "फर्जी बिक्री विलेख और संपत्ति हस्तांतरण धोखाधड़ी",
            "ta": "போலி விற்பனை ஆவணம் மற்றும் சொத்து மாற்ற மோசடி",
            "mr": "बनावट विक्रीपत्र आणि मालमत्ता हस्तांतरण फसवणूक",
            "bn": "জাল সেল ডিড ও সম্পত্তি হস্তান্তর জালিয়াতি",
            "te": "నకిలీ సేల్ డీడ్ మరియు ఆస్తి బదిలీ మోసం",
            "kn": "ನಕಲಿ ಮಾರಾಟ ಪತ್ರ ಮತ್ತು ಆಸ್ತಿ ಹಸ್ತಾಂತರ ವಂಚನೆ",
            "ml": "കള്ള സെയിൽ ഡീഡും സ്വത്ത് കൈമാറ്റ തട്ടിപ്പും",
            "pa": "ਜਾਲੀ ਵਿਕਰੀ ਦਸਤਾਵੇਜ਼ ਅਤੇ ਜਾਇਦਾਦ ਹਸਤਾਂਤਰਨ ਧੋਖਾਧੜੀ",
            "en": "forged sale deed and property transfer fraud",
        },
    },
    {
        "intent_id": "wrongful_dismissal_natural_justice",
        "relevant_cases": [
            {"case_id": "1424029", "relevance": 3},
            {"case_id": "63931940", "relevance": 2},
            {"case_id": "554611", "relevance": 1},
        ],
        "topics": {
            "gu": "વિભાગીય તપાસ વિના નોકરીમાંથી દૂર કરવું અને કુદરતી ન્યાય",
            "hi": "विभागीय जांच बिना नौकरी से हटाना और प्राकृतिक न्याय",
            "ta": "துறை விசாரணையின்றி பணிநீக்கம் மற்றும் இயற்கை நீதி",
            "mr": "विभागीय चौकशीशिवाय सेवेतून काढणे आणि नैसर्गिक न्याय",
            "bn": "বিভাগীয় তদন্ত ছাড়া চাকরি শেষ করা ও প্রাকৃতিক ন্যায়",
            "te": "శాఖాపర విచారణ లేకుండా ఉద్యోగం రద్దు చేయడం మరియు సహజ న్యాయం",
            "kn": "ವಿಭಾಗೀಯ ತನಿಖೆಯಿಲ್ಲದೆ ಸೇವೆಯಿಂದ ತೆಗೆದುಹಾಕುವುದು ಮತ್ತು ನೈಸರ್ಗಿಕ ನ್ಯಾಯ",
            "ml": "വിഭാഗീയ അന്വേഷണം കൂടാതെ സേവനം അവസാനിപ്പിക്കൽയും സ്വാഭാവിക നീതിയും",
            "pa": "ਵਿਭਾਗੀ ਜਾਂਚ ਬਿਨਾ ਸੇਵਾ ਖ਼ਤਮ ਕਰਨਾ ਅਤੇ ਕੁਦਰਤੀ ਇਨਸਾਫ਼",
            "en": "termination without departmental inquiry and natural justice",
        },
    },
    {
        "intent_id": "disability_public_bus_access",
        "relevant_cases": [
            {"case_id": "107199202", "relevance": 3},
            {"case_id": "150126499", "relevance": 2},
            {"case_id": "11902464", "relevance": 1},
        ],
        "topics": {
            "gu": "જાહેર બસમાં દિવ્યાંગ પ્રવેશ અધિકાર",
            "hi": "सार्वजनिक बस में दिव्यांग प्रवेश अधिकार",
            "ta": "பொது பேருந்தில் மாற்றுத்திறனாளி அணுகல் உரிமை",
            "mr": "सार्वजनिक बसमधील दिव्यांग प्रवेशाधिकार",
            "bn": "পাবলিক বাসে প্রতিবন্ধী প্রবেশাধিকার",
            "te": "పబ్లిక్ బస్సులో వికలాంగుల ప్రవేశ హక్కులు",
            "kn": "ಸಾರ್ವಜನಿಕ ಬಸ್ಸಿನಲ್ಲಿ ದಿವ್ಯಾಂಗ ಪ್ರವೇಶ ಹಕ್ಕು",
            "ml": "പൊതു ബസിലെ ഭിന്നശേഷിക്കാരുടെ പ്രവേശനാവകാശം",
            "pa": "ਪਬਲਿਕ ਬੱਸ ਵਿੱਚ ਵਿਸ਼ੇਸ਼ ਯੋਗਤਾ ਵਾਲਿਆਂ ਦੇ ਪ੍ਰਵੇਸ਼ ਅਧਿਕਾਰ",
            "en": "disability access rights in a public bus",
        },
    },
    {
        "intent_id": "murder_conviction_section_302",
        "relevant_cases": [
            {"case_id": "1054413", "relevance": 3},
            {"case_id": "1208137", "relevance": 2},
            {"case_id": "1054982", "relevance": 1},
        ],
        "topics": {
            "gu": "કલમ 302 હેઠળ હત્યાની સજા",
            "hi": "धारा 302 के तहत हत्या की सजा",
            "ta": "பிரிவு 302 கீழ் கொலை தண்டனை",
            "mr": "कलम 302 अंतर्गत खुनाची शिक्षा",
            "bn": "ধারা 302 এর অধীনে খুনের দণ্ড",
            "te": "సెక్షన్ 302 కింద హత్య శిక్ష",
            "kn": "ಸೆಕ್ಷನ್ 302 ಅಡಿಯಲ್ಲಿ ಕೊಲೆ ಶಿಕ್ಷೆ",
            "ml": "വിഭാഗം 302 പ്രകാരമുള്ള കൊലപാതക ശിക്ഷ",
            "pa": "ਧਾਰਾ 302 ਹੇਠ ਕਤਲ ਦੀ ਸਜ਼ਾ",
            "en": "murder conviction under section 302",
        },
    },
    {
        "intent_id": "motor_accident_compensation",
        "relevant_cases": [
            {"case_id": "165531402", "relevance": 3},
            {"case_id": "162283696", "relevance": 2},
            {"case_id": "11902464", "relevance": 1},
        ],
        "topics": {
            "gu": "મોટર અકસ્માતમાં ઇજા માટે વળતર",
            "hi": "मोटर दुर्घटना में चोट के लिए मुआवजा",
            "ta": "மோட்டார் விபத்து காயங்களுக்கு இழப்பீடு",
            "mr": "मोटार अपघातातील दुखापतीची नुकसानभरपाई",
            "bn": "মোটর দুর্ঘটনায় আঘাতের ক্ষতিপূরণ",
            "te": "మోటారు ప్రమాద గాయాలకు పరిహారం",
            "kn": "ಮೋಟಾರ್ ಅಪಘಾತ ಗಾಯ ಪರಿಹಾರ",
            "ml": "മോട്ടോർ അപകട പരിക്കിനുള്ള നഷ്ടപരിഹാരം",
            "pa": "ਮੋਟਰ ਹਾਦਸੇ ਦੀ ਚੋਟ ਲਈ ਮੁਆਵਜ਼ਾ",
            "en": "motor accident injury compensation",
        },
    },
    {
        "intent_id": "robbery_with_hurt",
        "relevant_cases": [
            {"case_id": "1380205", "relevance": 3},
            {"case_id": "164918331", "relevance": 2},
            {"case_id": "114927673", "relevance": 1},
        ],
        "topics": {
            "gu": "લૂંટ, મારપીટ અને રોકડ લૂંટ",
            "hi": "लूट, मारपीट और नकदी छीनना",
            "ta": "அடித்து காயப்படுத்தி பணம் பறித்த கொள்ளை",
            "mr": "मारहाण करून रोख रक्कम लुटणे",
            "bn": "আক্রমণ করে নগদ লুট",
            "te": "దాడి చేసి నగదు దోచుకున్న రాబరీ",
            "kn": "ಹಲ್ಲೆ ಮಾಡಿ ನಗದು ದೋಚಿದ ದರೋಡೆ",
            "ml": "അക്രമിച്ച് പണം കവർന്ന കവർച്ച",
            "pa": "ਮਾਰਪੀਟ ਕਰਕੇ ਨਕਦੀ ਲੁੱਟਣਾ",
            "en": "robbery with hurt and looting cash",
        },
    },
    {
        "intent_id": "rape_sexual_assault",
        "relevant_cases": [
            {"case_id": "31020514", "relevance": 3},
            {"case_id": "102868427", "relevance": 2},
            {"case_id": "28670053", "relevance": 1},
        ],
        "topics": {
            "gu": "બળાત્કાર અને યૌન શોષણ",
            "hi": "बलात्कार और यौन उत्पीड़न",
            "ta": "பலாத்காரம் மற்றும் பாலியல் துஷ்பிரயோகம்",
            "mr": "बलात्कार आणि लैंगिक अत्याचार",
            "bn": "ধর্ষণ ও যৌন নির্যাতন",
            "te": "బలాత్కారం మరియు లైంగిక దాడి",
            "kn": "ಬಲಾತ್ಕಾರ ಮತ್ತು ಲೈಂಗಿಕ ದೌರ್ಜನ್ಯ",
            "ml": "ബലാൽസംഗവും ലൈംഗിക അതിക്രമവും",
            "pa": "ਬਲਾਤਕਾਰ ਅਤੇ ਯੌਨ ਹਿੰਸਾ",
            "en": "rape and sexual assault",
        },
    },
    {
        "intent_id": "electricity_theft_meter_tampering",
        "relevant_cases": [
            {"case_id": "171690404", "relevance": 3},
            {"case_id": "1541399", "relevance": 2},
            {"case_id": "53105671", "relevance": 1},
        ],
        "topics": {
            "gu": "વીજળી ચોરી અને મીટર સાથે છેડછાડ",
            "hi": "बिजली चोरी और मीटर से छेड़छाड़",
            "ta": "மின்சார திருட்டு மற்றும் மீட்டர் சேதப்படுத்தல்",
            "mr": "वीज चोरी आणि मीटरमध्ये छेडछाड",
            "bn": "বিদ্যুৎ চুরি ও মিটার কারচুপি",
            "te": "విద్యుత్ చోరీ మరియు మీటర్ ట్యాంపరింగ్",
            "kn": "ವಿದ್ಯುತ್ ಕಳ್ಳತನ ಮತ್ತು ಮೀಟರ್ ತೊಂದರೆ",
            "ml": "വൈദ്യുതി മോഷണവും മീറ്റർ കൈകടത്തലും",
            "pa": "ਬਿਜਲੀ ਚੋਰੀ ਅਤੇ ਮੀਟਰ ਨਾਲ ਛੇੜਛਾੜ",
            "en": "electricity theft and meter tampering",
        },
    },
]


def main() -> None:
    queries: list[dict[str, object]] = []
    query_counter = 1

    for intent in INTENTS:
        for language in LANGUAGE_ORDER:
            topic = intent["topics"][language]
            for variant_index, template in enumerate(TEMPLATES[language], start=1):
                queries.append(
                    {
                        "query_id": f"Q{query_counter:03d}",
                        "language": language,
                        "query_text": template.format(topic=topic),
                        "relevant_cases": intent["relevant_cases"],
                        "intent_id": intent["intent_id"],
                        "variant_id": variant_index,
                    }
                )
                query_counter += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(queries, file, ensure_ascii=False, indent=2)

    print(f"Wrote {len(queries)} queries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
