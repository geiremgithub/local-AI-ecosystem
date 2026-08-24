def besvar_sporsmal(bruker_sporsmal):
    print("\n--- [RAG PROSESSPESIFIKASJON] ---")
    
    # 1. Tokenisering - dekoder eksplisitt for norsk visning
    tokens = ai_modell.tokenizer.tokenize(bruker_sporsmal)
    token_ids = ai_modell.tokenizer.encode(bruker_sporsmal, add_special_tokens=False)
    # Konverterer Qwen-spesifikke tegn (som Ġ) til lesbar tekst for loggen
    lesbare_tokens = [t.replace('Ġ', ' ').replace('Ã¥', 'å') for t in tokens]
    
    print(f"1. Tokenisering: Spørsmålet ble delt inn i {len(tokens)} tokens.")
    print(f"   - Tokens: {lesbare_tokens}")
    print(f"   - Token ID-er: {token_ids}")

    # 2. Embedding
    sporsmal_vektor = embedding_modell([bruker_sporsmal])[0]
    print(f"2. Embedding (Spørsmål-vektor): Totalt {len(sporsmal_vektor)} dimensjoner.")
    print(f"   - Vektortall (første 10): {[round(v, 4) for v in sporsmal_vektor[:10]]} ...")

    # 3. Databasesøk - Økt n_results for å sikre treff
    print("3. Databasesøk: Sorterer chunks basert på matchet avstand...")
    sok_resultat = db_samling.query(
        query_texts=[bruker_sporsmal], 
        n_results=5, 
        include=["documents", "metadatas", "distances", "embeddings"]
    )
    
    dokumenter = sok_resultat["documents"][0]
    metadataer = sok_resultat["metadatas"][0]
    avstander = sok_resultat["distances"][0]
    vektorer = sok_resultat["embeddings"][0]

    filtrerte_kilder = []
    print("   - Topp treff fra databasen:")
    for i in range(len(dokumenter)):
        print(f"     [{i}] Kilde: {metadataer[i]['kilde']} | Avstand: {avstander[i]:.4f}")
        filtrerte_kilder.append(f"Kilde: {metadataer[i]['kilde']}\nInnhold: {dokumenter[i]}")

    kombinert_kontekst = "\n\n".join(filtrerte_kilder)

    # 4. Forbedret System-instruks som hindrer bortforklaring
    meldinger = [
        {
            "role": "system", 
            "content": (
                "Du er en ekstremt nøyaktig assistent. Du skal KUN bruke informasjonen fra 'Relevante kilder'. "
                "Hvis svaret finnes i kildene, svar direkte. Ikke si 'informasjonen er ikke gitt' hvis kildene inneholder svaret. "
                "Hvis en kilde nevner et klokkeslett eller en dato for skolestart, oppgi dette eksplisitt."
            )
        },
        {
            "role": "user", 
            "content": f"Spørsmål: {bruker_sporsmal}\n\nRelevante kilder:\n{kombinert_kontekst}"
        }
    ]

    prompt = ai_modell.tokenizer.apply_chat_template(meldinger, tokenize=False, add_generation_prompt=True)

    start_tid = time.time()
    resultat = ai_modell(prompt, generation_config=generasjons_konfig, return_full_text=False)
    total_tid = time.time() - start_tid
    
    svar = resultat[0]["generated_text"].strip()
    genererte_tokens = len(ai_modell.tokenizer.encode(svar, add_special_tokens=False))
    
    print(f"\n-> Svar fra AI:\n{svar}")
    print(f"[Ytelse] Generering tok {total_tid:.2f} sekunder (ca. {genererte_tokens / total_tid:.1f} tokens/sek)\n")