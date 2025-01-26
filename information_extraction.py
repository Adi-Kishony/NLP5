import wikipedia
import spacy
import random
import gemini

# Set your Gemini API key
gemini.api_key = "your-api-key"

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

def evaluate_extractor(extractor, content, nlp):
    triplets = extractor(content, nlp)
    random_sample = random.sample(triplets, min(5, len(triplets)))
    print(f"\nNumber of triplets: {len(triplets)}")
    print("Random sample of triplets:")
    for triplet in random_sample:
        print(triplet)
    return triplets

def call_large_language_model(triplets):
    validated_triplets = []
    print("\nValidating triplets with LLM:")
    for triplet in triplets:
        prompt = f"Determine if the following triplet is valid based on general knowledge: Subject: {triplet[0]}, Relation: {triplet[1]}, Object: {triplet[2]}"
        try:
            response = gemini.Completion.create(
                prompt=prompt,
                max_tokens=50
            )
            is_valid = "yes" in response["text"].strip().lower()
            print(f"Triplet: {triplet}, Valid: {is_valid}")
            if is_valid:
                validated_triplets.append(triplet)
        except Exception as e:
            print(f"Error validating triplet {triplet}: {e}")
    return validated_triplets

def manual_verification(triplets):
    valid_count = 0
    print("\nManual Verification of Random Triplets:")
    for triplet in triplets:
        print(triplet)
        is_valid = input("Is this triplet valid? (yes/no): ").strip().lower()
        if is_valid == "yes":
            valid_count += 1
    print(f"Valid triplets: {valid_count}/{len(triplets)}")

def main():
    nlp = setup_nlp()

    # Define Wikipedia pages for analysis
    pages = ["Donald Trump", "Ruth Bader Ginsburg", "J. K. Rowling"]

    all_pos_triplets = []
    all_dep_triplets = []

    for page in pages:
        print(f"\nProcessing Wikipedia page: {page}")
        content = get_wikipedia_page_content(page)

        print("\nPOS Tag-Based Extractor Results:")
        pos_triplets = evaluate_extractor(pos_tag_based_extractor, content, nlp)
        all_pos_triplets.extend(pos_triplets)

        print("\nDependency Tree-Based Extractor Results:")
        dep_triplets = evaluate_extractor(dependency_tree_based_extractor, content, nlp)
        all_dep_triplets.extend(dep_triplets)

    # Validate triplets with LLM
    print("\nValidating POS Tag-Based Extractor Triplets with LLM")
    validated_pos_triplets = call_large_language_model(random.sample(all_pos_triplets, min(15, len(all_pos_triplets))))
    print(f"\nValidated POS Triplets: {validated_pos_triplets}")

    print("\nValidating Dependency Tree-Based Extractor Triplets with LLM")
    validated_dep_triplets = call_large_language_model(random.sample(all_dep_triplets, min(15, len(all_dep_triplets))))
    print(f"\nValidated Dependency Triplets: {validated_dep_triplets}")


# def main():
#     nlp = setup_nlp()
#
#     # Define Wikipedia pages for analysis
#     pages = ["Donald Trump", "Ruth Bader Ginsburg", "J. K. Rowling"]
#
#     for page in pages:
#         print(f"\nProcessing Wikipedia page: {page}")
#         content = get_wikipedia_page_content(page)
#
#         pos_results = pos_tag_based_extractor(content, nlp)
#         print(f"\nPOS Tag-Based Extractor Results: number of triplets {len(pos_results)}")
#         print(pos_results[:5])  # Display a few results
#
#         dep_results = dependency_tree_based_extractor(content, nlp)
#         print(f"\nDependency Tree-Based Extractor Results: number of triplets {len(dep_results)}")
#         print(dep_results[:5])  # Display a few results



if __name__ == "__main__":
    main()


