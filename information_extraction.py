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



def call_large_language_model(wiki_page, triplets, model_gemini):
    validated_triplets = []
    print("\nValidating triplets with LLM:")

    # Start a chat and send the Wikipedia page content once
    chat = model_gemini.start_chat(
        history=[
            {"role": "user", "parts": "Hello"},
            {"role": "model", "parts": "Great to meet you. What would you like to know?"},
        ]
    )

    chat.send_message(f"Here is the text for reference:\n{wiki_page}")

    batch_size = 10  # Validate in batches to reduce the number of API calls
    for i in range(7):
        batch = triplets[i*batch_size:i*batch_size+batch_size]
        triplets_text = "\n".join(
            [f"Triplet {idx+1}: Subject: {t[0]}, Relation: {t[1]}, Object: {t[2]}" for idx, t in enumerate(batch)]
        )
        try:
            response = chat.send_message(
                f"Determine (answer only yes or no) if the following relationship triplets are valid based on the supplied text:\n{triplets_text}",
                stream=True
            )

            # Collect and process response
            result = ""
            for chunk in response:
                result += chunk.text
            print(f"Batch Response: {result.strip()}")

            # Parse responses for each triplet
            responses = result.strip().split("\n")
            for triplet, res in zip(batch, responses):
                if "yes" in res.lower():
                    validated_triplets.append(triplet)

        except Exception as e:
            print(f"Quota exceeded. Skipping remaining triplets in this batch, {e}")
            break  # Exit loop if quota is exhausted

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
    validated_pos_triplets = call_large_language_model(text_and_triplets_pos[page][1], text_and_triplets_pos[page][0], model)
    print(f"\nValidated POS Triplets: {validated_pos_triplets}")


    print("\nValidating Dependency Tree-Based Extractor Triplets with LLM")
    validated_dep_triplets = call_large_language_model(text_and_triplets_pos[page][1], text_and_triplets_pos[page][0], model)
    print(f"\nValidated Dependency Triplets: {validated_dep_triplets}")

    # TODO: for each of the models give gemini the text and the triplets we found and tell it to identify relationship we didnt find - 2 calls


if __name__ == "__main__":
    main()
