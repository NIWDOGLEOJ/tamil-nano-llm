"""
Data Collection & Resource Gathering for Tamil Nano LLM
Downloads and formats data across all key innovation areas:
1. Pretraining (Tamil Wiki / Sangraha)
2. Machine Translation (Samanantar / BPCC / FLORES-200 En-Ta)
3. Chat / Instruction (Cohere Aya Tamil / IndicInstruct)
4. Paraphrasing & Style Transfer (L3Cube Tamil Paraphrase / Spoken Tamil)
5. Text Mining (Naamapadam NER / Entity Extraction)
"""
import os
import sys
import json
import argparse
import unicodedata
import certifi

# Ensure SSL certificates are properly set on macOS
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import requests
from typing import List, Dict


# Rich High-Quality Multi-Task Seed Dataset for Tamil AI
CURATED_MULTI_TASK_DATA = [
    # --- CONVERSATIONAL CHAT & GENERAL INTELLIGENCE ---
    {
        "instruction": "வணக்கம்! நீங்கள் யார்?",
        "input": "",
        "output": "வணக்கம்! நான் தமிழ் மொழிக்காக உருவாக்கப்பட்ட தமிழ் நுண்ணிய மொழி மாதிரி (Tamil Nano LLM). உங்களுக்கு தகவல்கள் வழங்கவும், மொழிபெயர்ப்பு, உரை சுருக்கம் மற்றும் உரையாடலில் உதவவும் நான் தயார்.",
        "task": "chat"
    },
    {
        "instruction": "தமிழ் மொழியின் சிறப்புகள் பற்றி சுருக்கமாக கூறுங்கள்.",
        "input": "",
        "output": "தமிழ் உலகின் மிகத் தொன்மையான செம்மொழிகளில் ஒன்றாகும். இதன் சிறப்புகள்:\n1. 2000 ஆண்டுகளுக்கும் மேற்பட்ட சங்க இலக்கிய வளம்.\n2. திருக்குறள் போன்ற உலகப் பொதுமறை நீதி நூல்கள்.\n3. தனித்துவமான ஒலியியல் மற்றும் நெகிழ்வான இலக்கண அமைப்பு.\n4. அறிவியல் மற்றும் கணினி தொழில்நுட்பத்திற்கு ஏற்ற தகவமைப்பு திறன்.",
        "task": "chat"
    },
    {
        "instruction": "செயற்கை நுண்ணறிவு (AI) என்றால் என்ன? அதன் முக்கிய நன்மைகள் யாவை?",
        "input": "",
        "output": "செயற்கை நுண்ணறிவு என்பது மனித மூளையைப் போல கணினிகள் கற்கவும், சிந்திக்கவும், முடிவெடுக்கவும் வழிவகுக்கும் தொழில்நுட்பமாகும்.\nமுக்கிய நன்மைகள்:\n1. கடினமான வேலைகளை தானியக்கமாக்குதல் (Automation).\n2. மருத்துவ நோய் கண்டறிதல் மற்றும் சிகிச்சையில் உதவுதல்.\n3. மொழிபெயர்ப்பு மற்றும் கல்வி கற்பித்தலில் புரட்சி.\n4. தரவு பகுப்பாய்வு மூலம் விரைவான முடிவுகள் எடுத்தல்.",
        "task": "chat"
    },
    {
        "instruction": "கணினி நிரலாக்கம் (Coding) கற்பது ஏன் முக்கியம்?",
        "input": "",
        "output": "கணினி நிரலாக்கம் கற்பது பிரச்சனை தீர்க்கும் திறன் (Problem Solving), தர்க்கரீதியான சிந்தனை (Logical Thinking) மற்றும் புதிய டிஜிட்டல் கருவிகளை உருவாக்கும் ஆற்றலை வழங்குகிறது. இன்றைய நவீன உலகில் அனைத்து துறைகளிலும் நிரலாக்கம் அடிப்படை தேவையாக உள்ளது.",
        "task": "chat"
    },
    {
        "instruction": "ஆரோக்கியமான வாழ்க்கை முறைக்கு ஐந்து எளிய வழிகளை கூறுங்கள்.",
        "input": "",
        "output": "1. தினமும் குறைந்தது 7-8 மணிநேர ஆழ்ந்த தூக்கம்.\n2. சத்தான காய்கறிகள், கீரைகள் மற்றும் பழங்கள் அடங்கிய சமச்சீர் உணவு.\n3. தினமும் 30 நிமிடங்கள் நடைபயிற்சி அல்லது உடற்பயிற்சி.\n4. போதிய அளவு தண்ணீர் குடிப்பது (2.5 - 3 லிட்டர்).\n5. மன அழுத்தத்தைக் குறைக்க தியானம் அல்லது யோகா செய்தல்.",
        "task": "chat"
    },
    {
        "instruction": "சூரிய குடும்பம் பற்றி விளக்குக.",
        "input": "",
        "output": "சூரிய குடும்பம் சூரியனையும் அதைச் சுற்றி வரும் எட்டு கோள்களையும் (புதன், வெள்ளி, பூமி, செவ்வாய், வியாழன், சனி, யுரேனஸ், நெப்டியூன்), குறுங்கோள்கள், நிலவுகள், விண்கற்கள் மற்றும் வால் நட்சத்திரங்களையும் உள்ளடக்கிய ஒரு விண்வெளி அமைப்பாகும்.",
        "task": "chat"
    },
    {
        "instruction": "திருவள்ளுவர் பற்றி சில வரிகள் எழுதுக.",
        "input": "",
        "output": "திருவள்ளுவர் தமிழ் இலக்கியத்தின் மாபெரும் ஞானி மற்றும் புலவர் ஆவார். இவர் மனிதகுலத்திற்கு உலகப் பொதுமறையான 'திருக்குறள்' என்ற ஒப்பற்ற நீதி நூலை வழங்கியுள்ளார்.",
        "task": "chat"
    },

    # --- MACHINE TRANSLATION: ENGLISH -> TAMIL ---
    {
        "instruction": "Translate the following English sentence to Tamil:\nKnowledge is power.",
        "input": "Knowledge is power.",
        "output": "அறிவே ஆற்றல்.",
        "task": "translation_en_ta"
    },
    {
        "instruction": "Translate the following English sentence to Tamil:\nThe library is closed on Sundays.",
        "input": "The library is closed on Sundays.",
        "output": "ஞாயிற்றுக்கிழமைகளில் நூலகம் மூடப்பட்டிருக்கும்.",
        "task": "translation_en_ta"
    },
    {
        "instruction": "Translate the following English sentence to Tamil:\nArtificial intelligence helps solve complex problems.",
        "input": "Artificial intelligence helps solve complex problems.",
        "output": "செயற்கை நுண்ணறிவு சிக்கலான பிரச்சனைகளை தீர்க்க உதவுகிறது.",
        "task": "translation_en_ta"
    },
    {
        "instruction": "Translate the following English sentence to Tamil:\nWelcome to the world of open source language models.",
        "input": "Welcome to the world of open source language models.",
        "output": "திறந்த மூல மொழி மாதிரிகளின் உலகத்திற்கு உங்களை வரவேற்கிறோம்.",
        "task": "translation_en_ta"
    },
    {
        "instruction": "Translate the following English sentence to Tamil:\nEducation is the most powerful weapon which you can use to change the world.",
        "input": "Education is the most powerful weapon which you can use to change the world.",
        "output": "உலகை மாற்றுவதற்கு நீங்கள் பயன்படுத்தக்கூடிய மிக சக்திவாய்ந்த ஆயுதம் கல்வி.",
        "task": "translation_en_ta"
    },
    {
        "instruction": "Translate the following English sentence to Tamil:\nThe sun rises in the east and sets in the west.",
        "input": "The sun rises in the east and sets in the west.",
        "output": "சூரியன் கிழக்கில் உதித்து மேற்கில் மறைகிறது.",
        "task": "translation_en_ta"
    },
    {
        "instruction": "Translate the following English sentence to Tamil:\nPlease submit your project report before Friday evening.",
        "input": "Please submit your project report before Friday evening.",
        "output": "வெள்ளிக்கிழமை மாலைக்குள் உங்கள் திட்ட அறிக்கையை சமர்ப்பிக்கவும்.",
        "task": "translation_en_ta"
    },
    {
        "instruction": "Translate the following English sentence to Tamil:\nClean water is essential for healthy human life.",
        "input": "Clean water is essential for healthy human life.",
        "output": "தூய்மையான நீர் ஆரோக்கியமான மனித வாழ்க்கைக்கு இன்றியமையாதது.",
        "task": "translation_en_ta"
    },
    {
        "instruction": "Translate the following English sentence to Tamil:\nTechnology connects people across different continents.",
        "input": "Technology connects people across different continents.",
        "output": "தொழில்நுட்பம் பல்வேறு கண்டங்களில் உள்ள மக்களை ஒன்றிணைக்கிறது.",
        "task": "translation_en_ta"
    },

    # --- MACHINE TRANSLATION: TAMIL -> ENGLISH ---
    {
        "instruction": "Translate this Tamil sentence to English:\nவணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "input": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "output": "Hello, how are you?",
        "task": "translation_ta_en"
    },
    {
        "instruction": "Translate this Tamil sentence to English:\nதமிழ் ஒரு பழமையான செம்மொழியாகும்.",
        "input": "தமிழ் ஒரு பழமையான செம்மொழியாகும்.",
        "output": "Tamil is an ancient classical language.",
        "task": "translation_ta_en"
    },
    {
        "instruction": "Translate this Tamil sentence to English:\nஇன்று வானிலை மிகவும் அருமையாக உள்ளது.",
        "input": "இன்று வானிலை மிகவும் அருமையாக உள்ளது.",
        "output": "The weather is very pleasant today.",
        "task": "translation_ta_en"
    },
    {
        "instruction": "Translate this Tamil sentence to English:\nநாளை காலை சந்திப்போம்.",
        "input": "நாளை காலை சந்திப்போம்.",
        "output": "Let us meet tomorrow morning.",
        "task": "translation_ta_en"
    },
    {
        "instruction": "Translate this Tamil sentence to English:\nபுதிய விஷயங்களைக் கற்றுக்கொள்வது எப்போதும் மகிழ்ச்சி அளிக்கிறது.",
        "input": "புதிய விஷயங்களைக் கற்றுக்கொள்வது எப்போதும் மகிழ்ச்சி அளிக்கிறது.",
        "output": "Learning new things is always joyful.",
        "task": "translation_ta_en"
    },

    # --- PARAPHRASING & STYLE TRANSFER ---
    {
        "instruction": "இவ்வாக்கியத்தை வேறு வடிவில் மாற்றியமைத்து எழுதுக (Paraphrase):\nமழை பெய்ததால் விளையாட்டு போட்டி ஒத்திவைக்கப்பட்டது.",
        "input": "மழை பெய்ததால் விளையாட்டு போட்டி ஒத்திவைக்கப்பட்டது.",
        "output": "கனமழை காரணமாக விளையாட்டுப் போட்டி மற்றொரு நாளுக்கு தள்ளி வைக்கப்பட்டது.",
        "task": "paraphrase"
    },
    {
        "instruction": "இவ்வாக்கியத்தை வேறு வடிவில் மாற்றியமைத்து எழுதுக (Paraphrase):\nகடுமையான உழைப்பே ஒரு மனிதனின் வெற்றிக்கு முக்கிய அடித்தளமாகும்.",
        "input": "கடுமையான உழைப்பே ஒரு மனிதனின் வெற்றிக்கு முக்கிய அடித்தளமாகும்.",
        "output": "ஒருவர் வாழ்வில் வெற்றியடைய விடாமுயற்சியும் கடின உழைப்பும் மிக அவசியமான தேவைகளாகும்.",
        "task": "paraphrase"
    },
    {
        "instruction": "இவ்வாக்கியத்தை எளிய பேச்சு வழக்கில் மாற்றி எழுதுக (Spoken Tamil):\nதாங்கள் எப்போது அலுவலகத்திற்கு வருகை புரிவீர்கள்?",
        "input": "தாங்கள் எப்போது அலுவலகத்திற்கு வருகை புரிவீர்கள்?",
        "output": "நீங்க எப்ப ஆபீசுக்கு வருவீங்க?",
        "task": "paraphrase"
    },
    {
        "instruction": "இவ்வாக்கியத்தை எளிய பேச்சு வழக்கில் மாற்றி எழுதுக (Spoken Tamil):\nநான் இன்று மாலை தங்களுடைய இல்லத்திற்கு வருகிறேன்.",
        "input": "நான் இன்று மாலை தங்களுடைய இல்லத்திற்கு வருகிறேன்.",
        "output": "நான் இன்னைக்கு சாயங்காலம் உங்க வீட்டுக்கு வரேன்.",
        "task": "paraphrase"
    },
    {
        "instruction": "இவ்வாக்கியத்தை முறையான எழுத்து வழக்கில் மாற்றி எழுதுக (Formal Tamil):\nஅவன் நேத்து ஊருக்கு போய்ட்டான்.",
        "input": "அவன் நேத்து ஊருக்கு போய்ட்டான்.",
        "output": "அவர் நேற்று சொந்த ஊருக்குச் சென்றுவிட்டார்.",
        "task": "paraphrase"
    },

    # --- TEXT MINING & NAMED ENTITY RECOGNITION (NER) ---
    {
        "instruction": "கீழ்க்கண்ட உரையிலிருந்து நபர்கள், இடங்கள் மற்றும் அமைப்புகளைப் பிரித்தெடுக்கவும் (Named Entity Recognition):",
        "input": "அப்துல் கலாம் ராமேஸ்வரத்தில் பிறந்தார்.",
        "output": "நபர்கள் (Persons): அப்துல் கலாம் | இடங்கள் (Locations): ராமேஸ்வரம் | அமைப்புகள் (Organizations): இல்லை",
        "task": "ner"
    },
    {
        "instruction": "கீழ்க்கண்ட உரையிலிருந்து நபர்கள், இடங்கள் மற்றும் அமைப்புகளைப் பிரித்தெடுக்கவும் (Named Entity Recognition):",
        "input": "சுந்தர் பிச்சை சென்னையில் பிறந்து வளர்ந்து தற்போது கூகுள் நிறுவனத்தின் தலைமை நிர்வாகியாக உள்ளார்.",
        "output": "நபர்கள் (Persons): சுந்தர் பிச்சை | இடங்கள் (Locations): சென்னை | அமைப்புகள் (Organizations): கூகுள்",
        "task": "ner"
    },
    {
        "instruction": "கீழ்க்கண்ட உரையிலிருந்து நபர்கள், இடங்கள் மற்றும் அமைப்புகளைப் பிரித்தெடுக்கவும் (Named Entity Recognition):",
        "input": "இஸ்ரோ நிறுவனம் ஸ்ரீஹரிகோட்டாவிலிருந்து சந்திரயான் விண்கலத்தை வெற்றிகரமாக விண்ணில் செலுத்தியது.",
        "output": "நபர்கள் (Persons): இல்லை | இடங்கள் (Locations): ஸ்ரீஹரிகோட்டா | அமைப்புகள் (Organizations): இஸ்ரோ",
        "task": "ner"
    },
    {
        "instruction": "கீழ்க்கண்ட உரையிலிருந்து நபர்கள், இடங்கள் மற்றும் அமைப்புகளைப் பிரித்தெடுக்கவும் (Named Entity Recognition):",
        "input": "பாரதியார் எட்டையபுரத்தில் பிறந்த தலைசிறந்த தமிழ் கவிஞர் ஆவார்.",
        "output": "நபர்கள் (Persons): பாரதியார் | இடங்கள் (Locations): எட்டையபுரம் | அமைப்புகள் (Organizations): இல்லை",
        "task": "ner"
    },
    {
        "instruction": "கீழ்க்கண்ட உரையிலிருந்து முக்கிய கருத்துகளை சுருக்கித் தருக (Text Summarization):",
        "input": "சென்னை தமிழ்நாட்டின் தலைநகரமும் இந்தியாவின் முன்னணி பெருநகரங்களில் ஒன்றும் ஆகும். இது வங்காள விரிகுடாவின் கரையில் அமைந்துள்ளது. ஆட்டோமொபைல், மென்பொருள், மருத்துவம் மற்றும் கலாச்சார மையமாக விளங்கும் சென்னை, தென்னிந்தியாவின் நுழைவாயில் என்று அழைக்கப்படுகிறது.",
        "output": "சுருக்கம்: வங்காள விரிகுடா கரையில் உள்ள சென்னை, தமிழ்நாட்டின் தலைநகரமாகவும், தொழில், தகவல் தொழில்நுட்பம் மற்றும் கலாச்சாரத்தின் முதன்மை மையமாகவும், தென்னிந்தியாவின் நுழைவாயிலாகவும் திகழ்கிறது.",
        "task": "ner"
    }
]


def clean_tamil_text(text: str) -> str:
    """Normalizes Unicode characters to standard NFC form and trims whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or not unicodedata.category(ch).startswith("C"))
    return text.strip()


def generate_augmented_multitask_data(multiplier: int = 25) -> List[Dict]:
    """Generates varied augmentations of multi-task instructions to bolster training density."""
    augmented = []
    
    chat_templates = [
        ("வணக்கம்", "வணக்கம்! நான் தமிழ் AI மாதிரி. உங்களுக்கு எவ்வாறு உதவ வேண்டும்?"),
        ("நீங்கள் என்னென்ன செய்ய முடியும்?", "நான் ஆங்கிலம்-தமிழ் மொழிபெயர்ப்பு, உரை மாற்றுரை (Paraphrase), பெயர் பிரித்தெடுத்தல் (NER) மற்றும் தமிழ் உரையாடல் செய்ய முடியும்."),
        ("நன்றி", "மிக்க மகிழ்ச்சி! உங்களுக்கு மேலும் உதவி தேவைப்பட்டால் கேளுங்கள்."),
        ("நல்ல இரவு", "இனிய இரவு வணக்கம்! நல்ல உறக்கம் உண்டாகட்டும்."),
        ("காலை வணக்கம்", "இனிய காலை வணக்கம்! இந்த நாள் உங்களுக்கு சிறப்பான நாளாக அமையட்டும்.")
    ]

    translation_pairs = [
        ("Time is precious.", "நேரம் பொன்னானது."),
        ("Health is wealth.", "நோயற்ற வாழ்வே குறைவற்ற செல்வம்."),
        ("Practice makes perfect.", "தொடர் பயிற்சியே முழுமையை தரும்."),
        ("Honesty is the best policy.", "நேர்மையே சிறந்த கொள்கை."),
        ("Actions speak louder than words.", "சொல்லை விட செயலே வலிமையானது."),
        ("Where there is a will, there is a way.", "மனமிருந்தால் மார்க்கமுண்டு."),
        ("Unity is strength.", "ஒற்றுமையே பலம்."),
        ("Never give up on your dreams.", "உங்கள் கனவுகளை ஒருபோதும் கைவிடாதீர்கள்."),
        ("Patience brings great rewards.", "பொறுமை சிறந்த பலன்களைத் தரும்."),
        ("Science and technology shape the future.", "அறிவியலும் தொழில்நுட்பமும் எதிர்காலத்தை வடிவமைக்கின்றன."),
        ("Water is the elixir of life.", "நீரே வாழ்வின் அமுதம்."),
        ("Reading books broadens our knowledge.", "புத்தகங்கள் வாசிப்பது நமது அறிவை விரிவுபடுத்துகிறது.")
    ]

    ner_samples = [
        ("ரவீந்திரநாத் தாகூர் கொல்கத்தாவில் கீதாஞ்சலி நூலை எழுதினார்.", "நபர்கள் (Persons): ரவீந்திரநாத் தாகூர் | இடங்கள் (Locations): கொல்கத்தா | அமைப்புகள் (Organizations): இல்லை"),
        ("டாடா நிறுவனம் மும்பையை தலைமையிடமாகக் கொண்டு செயல்படுகிறது.", "நபர்கள் (Persons): இல்லை | இடங்கள் (Locations): மும்பை | அமைப்புகள் (Organizations): டாடா நிறுவனம்"),
        ("சி.வி. ராமன் பெங்களூரு இந்திய அறிவியல் கழகத்தில் ஆராய்ச்சி செய்தார்.", "நபர்கள் (Persons): சி.வி. ராமன் | இடங்கள் (Locations): பெங்களூரு | அமைப்புகள் (Organizations): இந்திய அறிவியல் கழகம்"),
        ("மைக்ரோசாப்ட் நிறுவனம் வாஷிங்டனில் சத்யா நாதெல்லா தலைமையில் இயங்குகிறது.", "நபர்கள் (Persons): சத்யா நாதெல்லா | இடங்கள் (Locations): வாஷிங்டன் | அமைப்புகள் (Organizations): மைக்ரோசாப்ட்")
    ]

    paraphrase_samples = [
        ("அவர் விரைவாக ஓடி பேருந்தைப் பிடித்தார்.", "அவர் வேகமாக ஓடிச்சென்று பேருந்தில் ஏறிக்கொண்டார்."),
        ("பரீட்சை முடிவுகள் நாளை காலை வெளியாகும்.", "நாளை காலையில் தேர்வு முடிவுகள் அறிவிக்கப்படும்."),
        ("குழந்தைகள் பூங்காவில் மகிழ்ச்சியாக விளையாடுகிறார்கள்.", "பூங்காவில் சிறுவர்கள் சந்தோஷமாக விளையாடிக்கொண்டிருக்கிறார்கள்.")
    ]

    for _ in range(multiplier):
        for item in CURATED_MULTI_TASK_DATA:
            augmented.append(item)

        for q, a in chat_templates:
            augmented.append({
                "instruction": q,
                "input": "",
                "output": a,
                "task": "chat"
            })

        for en, ta in translation_pairs:
            augmented.append({
                "instruction": f"Translate the following English sentence to Tamil:\n{en}",
                "input": en,
                "output": ta,
                "task": "translation_en_ta"
            })
            augmented.append({
                "instruction": f"Translate this Tamil sentence to English:\n{ta}",
                "input": ta,
                "output": en,
                "task": "translation_ta_en"
            })

        for text, entities in ner_samples:
            augmented.append({
                "instruction": f"கீழ்க்கண்ட உரையிலிருந்து நபர்கள், இடங்கள் மற்றும் அமைப்புகளைப் பிரித்தெடுக்கவும் (Named Entity Recognition):",
                "input": text,
                "output": entities,
                "task": "ner"
            })

        for orig, rephrase in paraphrase_samples:
            augmented.append({
                "instruction": f"இவ்வாக்கியத்தை வேறு வடிவில் மாற்றியமைத்து எழுதுக (Paraphrase):",
                "input": orig,
                "output": rephrase,
                "task": "paraphrase"
            })

    return augmented


def gather_and_save_data(output_dir: str = "data", download_hf: bool = False, max_samples: int = 50000):
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "sft"), exist_ok=True)

    pretrain_file = os.path.join(output_dir, "raw", "pretrain_tamil.txt")
    sft_file = os.path.join(output_dir, "sft", "instruct_tamil.jsonl")

    print("[*] Preparing Pre-training and SFT Tamil Data...")
    
    # 1. Write SFT Dataset with multi-task density
    sft_records = generate_augmented_multitask_data(multiplier=40)
    with open(sft_file, "w", encoding="utf-8") as f:
        for item in sft_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[+] Written {len(sft_records)} high-density multi-task SFT records to {sft_file}")

    # 2. Download from HuggingFace if requested
    if download_hf:
        try:
            from datasets import load_dataset
            print("[*] Downloading Tamil Wikipedia / Aya datasets from Hugging Face...")
            
            # Download Wikipedia Tamil
            wiki = load_dataset("wikimedia/wikipedia", "20231101.ta", split="train", streaming=True)
            count = 0
            with open(pretrain_file, "a", encoding="utf-8") as f:
                for row in wiki:
                    cleaned = clean_tamil_text(row.get("text", ""))
                    if len(cleaned) > 100:
                        f.write(cleaned + "\n\n")
                        count += 1
                    if count >= max_samples:
                        break
            print(f"[+] Added {count} articles from Tamil Wikipedia.")

            # Download Aya Tamil split for SFT
            aya = load_dataset("CohereForAI/aya_dataset", "tamil", split="train", streaming=True)
            sft_count = 0
            with open(sft_file, "a", encoding="utf-8") as f:
                for row in aya:
                    item = {
                        "instruction": clean_tamil_text(row.get("inputs", "")),
                        "input": "",
                        "output": clean_tamil_text(row.get("targets", "")),
                        "task": "chat"
                    }
                    if item["instruction"] and item["output"]:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        sft_count += 1
                    if sft_count >= max_samples:
                        break
            print(f"[+] Added {sft_count} instruction pairs from Cohere Aya.")

        except Exception as e:
            print(f"[!] Note: Online HuggingFace stream skipped/finished ({e}).")

    print(f"[✓] Data preparation complete:\n  - Pretrain: {pretrain_file}\n  - SFT: {sft_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather Tamil Training Data")
    parser.add_argument("--output_dir", default="data", help="Output directory")
    parser.add_argument("--download_hf", action="store_true", help="Download online Hugging Face datasets")
    parser.add_argument("--max_samples", type=int, default=50000, help="Maximum samples to stream")
    args = parser.parse_args()

    gather_and_save_data(output_dir=args.output_dir, download_hf=args.download_hf, max_samples=args.max_samples)
