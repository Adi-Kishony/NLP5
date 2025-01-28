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


def get_triplets_text(triplets):
    return "\n".join(
        [
            f"Triplet {idx + 1}: Subject: {t[0]}, Relation: {t[1]}, Object: {t[2]}"
            for idx, t in enumerate(triplets)]
    )


def evaluate_model_precision(wiki_page, triplets, model_gemini):
    validated_triplets = []
    print("\nValidating triplets with LLM:")

    # Start a chat and send the Wikipedia page content once
    chat = model_gemini.start_chat(
        history=[
            {"role": "user", "parts": "Hello"},
            {"role": "model",
             "parts": "Great to meet you. What would you like to know?"},
        ]
    )

    chat.send_message(f"Here is the text for reference:\n{wiki_page}")
    num_triplets_validated = 0
    batch_size = 12  # Validate in batches to reduce the number of API calls
    for i in range(6):
        if i * batch_size + batch_size >= len(triplets):
            break
        batch = triplets[i * batch_size:i * batch_size + batch_size]
        triplets_text = get_triplets_text(batch)
        try:
            response = chat.send_message(
                f"Determine (answer only yes or no) if the following "
                f"relationship triplets are valid based on the supplied text:\n{triplets_text}",
                stream=True
            )
            print(f"\nBatch triplets:\n {triplets_text}")
            # Collect and process response
            result = ""
            for chunk in response:
                result += chunk.text
            print(f"\nBatch Response:\n {result.strip()}")

            # Parse responses for each triplet
            responses = result.strip().split("\n")
            for triplet, res in zip(batch, responses):
                num_triplets_validated += 1
                if "yes" in res.lower():
                    validated_triplets.append(triplet)

        except Exception as e:
            print(
                f"Quota exceeded. Skipping remaining triplets in this batch, {e}")
            break  # Exit loop if quota is exhausted
    print(f"Percentage of correctly extracted triplets: {len(validated_triplets) / num_triplets_validated * 100:.2f}%")
    return validated_triplets


def evaluate_model_recall(wiki_page, triplets, model_gemini):
    prompt = f"Given this wikipedia page, and the following relationship " \
             f"triplets, can you provide only the relationships in the " \
             f"text that we did not find, for each one display it in a separate" \
             f" line wiki page mention only relations that appear in the text:" \
             f"\n Text: {wiki_page}\nTriplets: {get_triplets_text(triplets)}\n"

    response = model_gemini.generate_content(prompt)
    missed_relations = response.text.count('\n')
    print(f"Percentage of relations we missed: {missed_relations / (len(triplets) + missed_relations) * 100:.2f}%")
    print(response.text)


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
        # print(dep_results[:5])  # Display a few results
        text_and_triplets_tree[page] = (dep_results, content)

    # Validate triplets with LLM
    genai.configure(api_key="AIzaSyCE8xkfcHo7BPBrYRBmtzcw4Sb8bXngS2Q")
    model = genai.GenerativeModel("gemini-1.5-flash")

    page = "Donald Trump"

    # test percentage correctly extracted triplets relation:
    print(f"\nTesting Precision POS Tag-Based Extractor Triplets with LLM "
          f"of page {page}")
    validated_pos_triplets = evaluate_model_precision(
        text_and_triplets_pos[page][1], text_and_triplets_pos[page][0], model)

    print(f"\nTesting Precision Tree-Based Extractor Triplets with LLM of page {page}")
    validated_dep_triplets = evaluate_model_precision(
        text_and_triplets_pos[page][1], text_and_triplets_pos[page][0], model)

    # test percentage relations we missed
    print(f"\nTesting Recall POS Tag-Based Extractor missed Triplets with LLM page {page}")
    evaluate_model_recall(
        text_and_triplets_pos[page][1], text_and_triplets_pos[page][0],
        model)

    print(
        f"\nTesting Recall Tree-Based Extractor missed Triplets with LLM page {page}")
    evaluate_model_recall(
        text_and_triplets_pos[page][1], text_and_triplets_pos[page][0], model)


if __name__ == "__main__":
    main()
