import spacy
import wikipedia

def extract_pos_triplets(page_content):
    """
    Extract triplets based on POS tags.
    :param page_content: Text content from a Wikipedia page.
    :return: List of triplets (Subject, Relation, Object).
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(page_content)
    triplets = []

    # Iterate through the tokens to find proper noun sequences
    proper_nouns = []
    for token in doc:
        if token.pos_ == "PROPN":
            proper_nouns.append(token)
        else:
            if proper_nouns:
                proper_nouns.append(None)  # Mark sequence break

    # Find triplets based on the described rules
    for i in range(len(proper_nouns) - 1):
        if proper_nouns[i] and proper_nouns[i + 1]:  # Two consecutive PROPN sequences
            subject = proper_nouns[i].text
            obj = proper_nouns[i + 1].text

            # Extract the relation between them
            relation_tokens = []
            for tok in doc[proper_nouns[i].i + 1 : proper_nouns[i + 1].i]:
                if tok.pos_ in {"VERB", "ADP"}:  # Include verbs and prepositions
                    relation_tokens.append(tok.text)

            if relation_tokens:
                relation = " ".join(relation_tokens)
                triplets.append((subject, relation, obj))

    return triplets

def extract_dependency_triplets(page_content):
    """
    Extract triplets based on dependency trees.
    :param page_content: Text content from a Wikipedia page.
    :return: List of triplets (Subject, Relation, Object).
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(page_content)
    triplets = []

    for token in doc:
        # Identify proper noun heads (tokens with PROPN not part of a compound relation)
        if token.pos_ == "PROPN" and token.dep_ != "compound":
            subject_set = {child.text for child in token.children if child.dep_ == "compound"}
            subject_set.add(token.text)
            subject = " ".join(sorted(subject_set))

            for child in token.children:
                if child.dep_ == "nsubj":
                    h = child.head
                    for obj in h.children:
                        if obj.dep_ == "dobj":
                            # Condition 1: Subject -> nsubj -> Head -> dobj -> Object
                            object_set = {obj.text}
                            object_set.update(child.text for child in obj.children if child.dep_ == "compound")
                            obj_text = " ".join(sorted(object_set))
                            triplets.append((subject, h.text, obj_text))

                        elif obj.dep_ == "prep":
                            # Condition 2: Subject -> nsubj -> Head -> prep -> pobj
                            pobj = next((child for child in obj.children if child.dep_ == "pobj"), None)
                            if pobj:
                                object_set = {pobj.text}
                                object_set.update(child.text for child in pobj.children if child.dep_ == "compound")
                                obj_text = " ".join(sorted(object_set))
                                relation = f"{h.text} {obj.text}"
                                triplets.append((subject, relation, obj_text))

    return triplets

def main():
    # Example Wikipedia pages
    pages = ["Donald Trump", "Ruth Bader Ginsburg", "J. K. Rowling"]
    for page in pages:
        print(f"Processing page: {page}")
        content = wikipedia.page(page).content

        # Extract triplets based on POS tags
        pos_triplets = extract_pos_triplets(content)
        print(f"POS-based triplets for {page}:\n", pos_triplets)

        # Extract triplets based on dependency trees
        dep_triplets = extract_dependency_triplets(content)
        print(f"Dependency-based triplets for {page}:\n", dep_triplets)

if __name__ == "__main__":
    main()

import wikipedia
import spacy

def setup_nlp():
    # Load SpaCy model
    return spacy.load("en_core_web_sm")

def get_wikipedia_page_content(page_title):
    # Retrieve content from a Wikipedia page
    return wikipedia.page(page_title).content

def pos_tag_based_extractor(text, nlp):
    doc = nlp(text)
    triplets = []

    # Iterate through tokens to find proper nouns
    for i, token in enumerate(doc):
        if token.pos_ == "PROPN":
            subject = token.text
            for j in range(i + 1, len(doc)):
                if doc[j].pos_ == "PROPN":
                    # Check for intermediate tokens
                    relation_tokens = [t for t in doc[i+1:j] if t.pos_ not in ["PUNCT"]]
                    if any(t.pos_ == "VERB" for t in relation_tokens):
                        relation = " ".join([t.text for t in relation_tokens if t.pos_ in ["VERB", "ADP"]])
                        obj = doc[j].text
                        triplets.append((subject, relation, obj))
                        break
    return triplets

def dependency_tree_based_extractor(text, nlp):
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


