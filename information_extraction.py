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

