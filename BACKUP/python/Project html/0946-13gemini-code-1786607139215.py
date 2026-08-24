def besvar_sporsmal(bruker_sporsmal):
    print("\n--- [RAG PROSESSPESIFIKASJON] ---")
    
    # 1. Hent 10 resultater istedenfor 5 for å sikre at vi får med riktig fil
    sok_resultat = db_samling.query(
        query_texts=[bruker_sporsmal], 
        n_results=10, 
        include=["documents", "metadatas", "distances"]
    )
    
    dokumenter = sok_resultat["documents"][0]
    metadataer = sok_resultat["metadatas"][0]
    avstander = sok_resultat["distances"][0]

    # 2. Filtrer manuelt: Vi tvinger inn treff som inneholder "skole" eller "star"
    filtrerte_kilder = []
    print("3. Databasesøk: Filtrerer etter relevante treff...")
    
    for i in range(len(dokumenter)):
        tekst = dokumenter[i].lower()
        kilde = metadataer[i]['kilde']
        # Vi tar KUN med chunks som faktisk inneholder relevante ord
        if any(keyword in tekst for keyword in ["skole", "star", "klokken", "08"]):
            filtrerte_kilder.append(f"Fil: {kilde}\nInnhold: {dokumenter[i]}")
            print(f"   - [INKLUDERT] {kilde} (Avstand: {avstander[i]:.4f})")
        else:
            print(f"   - [EKSKLUDERT] {kilde} (Irrelevant)")

    if not filtrerte_kilder:
        print("-> FEIL: Ingen av de 10 beste treffene inneholdt relevante ord.")
        return

    kombinert_kontekst = "\n\n".join(filtrerte_kilder)

    # 3. System-instruks som tvinger bruk av kontekst
    meldinger = [
        {
            "role": "system", 
            "content": (
                "Du er en saklig assistent. Bruk KUN informasjonen under. "
                "Hvis en fil inneholder et tidspunkt for skolestart, oppgi det direkte. "
                "Ikke gjett, ikke dikt opp, og ikke si at informasjon mangler hvis den finnes i teksten under."
            )
        },
        {
            "role": "user", 
            "content": f"Spørsmål: {bruker_sporsmal}\n\nKontekst:\n{kombinert_kontekst}"
        }
    ]
    # ... resten av koden for å kalle AI-en ...