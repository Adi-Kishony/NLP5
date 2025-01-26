import wikipedia
import spacy
import random
import google.generativeai as genai



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
    # TODO - verfify this
    doc = nlp(text)
    triplets = []

    # Find proper noun heads
    proper_nouns = {}
    for token in doc:
        if token.pos_ == "PROPN" and token.dep_ != "compound":
            proper_nouns[token] = {t.text for t in token.children if
                                   t.dep_ == "compound"}.union({token.text})

    # Extract triplets based on conditions
    for h1, pn1 in proper_nouns.items():
        for h2, pn2 in proper_nouns.items():
            if h1 != h2:
                # Condition 1
                if h1.head == h2.head and h1.dep_ == "nsubj" and h2.dep_ == "dobj":
                    triplets.append(
                        (" ".join(pn1), h1.head.text, " ".join(pn2)))
                # Condition 2
                elif h1.head == h2.head.head and h1.dep_ == "nsubj" and h2.head.dep_ == "prep" and h2.dep_ == "pobj":
                    triplets.append((" ".join(pn1),
                                     f"{h1.head.text} {h2.head.text}",
                                     " ".join(pn2)))
    return triplets



def call_large_language_model(text_and_triplets, model_gemini):
    validated_triplets = []
    print("\nValidating triplets with LLM:")
    for page, content in text_and_triplets.items():
        triplets, wiki_page = content
        for triplet in triplets:
            prompt = f"Determine (answer only yesy or no) if the following relationship triplet is valid based on the supplied text:\n text: {wiki_page}\n," \
                     f" Subject: {triplet[0]} Relation: {triplet[1]}, Object: {triplet[2]}"
            try:
                response = model_gemini.generate_content(prompt)
                print(response.text)
                is_valid = "yes" in response["text"].strip().lower()
                print(f"Triplet: {triplet}, Valid: {is_valid}")
                if is_valid:
                    validated_triplets.append(triplet)
            except Exception as e:
                print(f"Error validating triplet {triplet}: {e}")
    print(f"percentage of validated triplets: {len(validated_triplets)/len(triplets)*100}%")
    return validated_triplets



def main():
    nlp = setup_nlp()

    # Define Wikipedia pages for analysis
    pages = ["Donald Trump", "Ruth Bader Ginsburg", "J. K. Rowling"]

    text_and_triplets_pos = dict()
    text_and_triplets_tree = dict()

    for page in pages:
        print(f"\nProcessing Wikipedia page: {page}")
        content = get_wikipedia_page_content(page)

        pos_results = pos_tag_based_extractor(content, nlp)
        print(
            f"\nPOS Tag-Based Extractor Results: number of triplets {len(pos_results)}")
        print(random.sample(pos_results, 5))  # Display a few random results
        # print(pos_results[:5])  # Display a few results
        text_and_triplets_pos[page] = (pos_results, content)

        dep_results = dependency_tree_based_extractor(content, nlp)
        print(
            f"\nDependency Tree-Based Extractor Results: number of triplets {len(dep_results)}")
        print(random.sample(dep_results, 5))  # Display a few random results
        #print(dep_results[:5])  # Display a few results
        text_and_triplets_tree[page] = (dep_results, content)

    # Validate triplets with LLM
    genai.configure(api_key="AIzaSyCE8xkfcHo7BPBrYRBmtzcw4Sb8bXngS2Q")
    model = genai.GenerativeModel("gemini-1.5-flash")
    print("\nValidating POS Tag-Based Extractor Triplets with LLM")
    validated_pos_triplets = call_large_language_model(text_and_triplets_pos, model)
    print(f"\nValidated POS Triplets: {validated_pos_triplets}")

    print("\nValidating Dependency Tree-Based Extractor Triplets with LLM")
    validated_dep_triplets = call_large_language_model(text_and_triplets_tree, model)
    print(f"\nValidated Dependency Triplets: {validated_dep_triplets}")


if __name__ == "__main__":
    main()
