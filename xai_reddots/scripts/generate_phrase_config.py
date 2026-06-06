import os, glob

def count_syllables(phonemes):
    vowels = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
    count = sum(1 for p in phonemes if ''.join([c for c in p if not c.isdigit()]) in vowels)
    return max(1, count)

phrases = [
    "ok_google",
    "there_s_no_such_thing_as_a_free_lunch",
    "a_watched_pot_never_boils",
    "jealousy_has_twenty_twenty_vision",
    "necessity_is_the_mother_of_invention",
    "artificial_intelligence_is_for_real",
    "only_lawyers_love_millionaires",
    "birthday_parties_have_cupcakes_and_ice_cream"
]

displays = {
    "ok_google": "Ok Google",
    "there_s_no_such_thing_as_a_free_lunch": "There's no such thing as a free lunch",
    "a_watched_pot_never_boils": "A watched pot never boils",
    "jealousy_has_twenty_twenty_vision": "Jealousy has twenty twenty vision",
    "necessity_is_the_mother_of_invention": "Necessity is the mother of invention",
    "artificial_intelligence_is_for_real": "Artificial intelligence is for real",
    "only_lawyers_love_millionaires": "Only lawyers love millionaires",
    "birthday_parties_have_cupcakes_and_ice_cream": "Birthday parties have cupcakes and ice cream"
}

output = []

for phrase in phrases:
    tg_files = glob.glob(f"xai_reddots/alignments/{phrase}/*.TextGrid")
    if not tg_files:
        print(f"Skipping {phrase}")
        continue
    
    with open(tg_files[0], 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        
    words_data = []
    
    # Simple parse for words and phones
    # We'll just find all words and all phones, then match by time
    items = []
    curr_class = None
    curr_min = 0.0
    curr_max = 0.0
    curr_text = ""
    
    tier_name = ""
    for line in lines:
        if 'name = "words"' in line: tier_name = 'words'
        elif 'name = "phones"' in line: tier_name = 'phones'
        
        if line.startswith('xmin ='): curr_min = float(line.split('=')[1].strip())
        elif line.startswith('xmax ='): curr_max = float(line.split('=')[1].strip())
        elif line.startswith('text ='):
            curr_text = line.split('=')[1].strip().strip('"')
            if curr_text and curr_text not in ['sp', 'sil']:
                items.append((tier_name, curr_min, curr_max, curr_text))
                
    words = [i for i in items if i[0] == 'words']
    phones = [i for i in items if i[0] == 'phones']
    
    syl_weights = []
    w_list = []
    
    for w in words:
        w_text = w[3]
        w_min = w[1]
        w_max = w[2]
        w_phones = []
        for p in phones:
            p_min = p[1]
            p_max = p[2]
            p_text = p[3]
            if p_min >= w_min - 0.01 and p_max <= w_max + 0.01:
                p_clean = ''.join([c for c in p_text if not c.isdigit()])
                w_phones.append(p_clean)
        w_list.append((w_text, w_phones))
        syl_weights.append(count_syllables(w_phones))
        
    output.append("    {")
    output.append(f"        'key'    : '{phrase}',")
    output.append(f"        'display': \"{displays[phrase]}\",")
    output.append("        'words'  : [")
    for w, phs in w_list:
        output.append(f"            ('{w}', {phs}),")
    output.append("        ],")
    output.append(f"        'syl_weights': {syl_weights},")
    output.append("    },")

with open('xai_reddots/scripts/phrase_configs.txt', 'w') as f:
    f.write("\n".join(output))
print("Done writing configs!")
