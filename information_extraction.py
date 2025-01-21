import wikipedia
import spacy

def setup_nlp():
    # Load SpaCy model
    return spacy.load("en_core_web_sm")

def get_wikipedia_page_content(page_title):
    # Retrieve content from a Wikipedia page
    return wikipedia.page(page_title, auto_suggest=False).content

def pos_tag_based_extractor(text, nlp):
    doc = nlp(text)
    triplets = []

    # Iterate through tokens to find proper nouns
    for i, token in enumerate(doc):
        if token.pos_ == "PROPN":
            subject = token.text
            for j in range(i + 1, len(doc)):
                if doc[j].pos_ == "PUNCT":
                    break
                if doc[j].pos_ == "PROPN":
                    contains_verb = False
                    relation_tokens = []
                    for k in range(i + 1, j):
                        if doc[k].pos_ == "VERB":
                            contains_verb = True
                        if doc[k].pos_ in ["VERB", "ADP"]:
                            relation_tokens.append(doc[k])
                    relation = " ".join([t.text for t in relation_tokens])
                    if contains_verb:
                        obj = doc[j].text
                        triplets.append((subject, relation, obj))
                        break
    return triplets

def dependency_tree_based_extractor(text, nlp):
    #TODO - verfify this
    doc = nlp(text)
    triplets = []

    # Find proper noun heads
    proper_nouns = {}
    for token in doc:
        if token.pos_ == "PROPN" and token.dep_ != "compound":
            proper_nouns[token] = {t.text for t in token.children if t.dep_ == "compound"}.union({token.text})

    # Extract triplets based on conditions
    for h1, pn1 in proper_nouns.items():
        for h2, pn2 in proper_nouns.items():
            if h1 != h2:
                # Condition 1
                if h1.head == h2.head and h1.dep_ == "nsubj" and h2.dep_ == "dobj":
                    triplets.append((" ".join(pn1), h1.head.text, " ".join(pn2)))
                # Condition 2
                elif h1.head == h2.head.head and h1.dep_ == "nsubj" and h2.head.dep_ == "prep" and h2.dep_ == "pobj":
                    triplets.append((" ".join(pn1), f"{h1.head.text} {h2.head.text}", " ".join(pn2)))
    return triplets

def main():
    nlp = setup_nlp()

    # Define Wikipedia pages for analysis
    pages = ["Donald Trump", "Ruth Bader Ginsburg", "J. K. Rowling"]

    for page in pages:
        print(f"Processing Wikipedia page: {page}")
        content = get_wikipedia_page_content(page)

        print("\nPOS Tag-Based Extractor Results:")
        pos_results = pos_tag_based_extractor(content, nlp)
        print(pos_results[:5])  # Display a few results

        print("\nDependency Tree-Based Extractor Results:")
        dep_results = dependency_tree_based_extractor(content, nlp)
        print(dep_results[:5])  # Display a few results

if __name__ == "__main__":
    main()


