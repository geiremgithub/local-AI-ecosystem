SELECT status,personid,Medlemsnavn,Epost,Fakturabeløp,Fakturadato,Forfallsdato,Produkt,Produkttype,Restbeløp,Antall_purringer,Fakturamottaker_Navn,Fakturamottaker_Epost
FROM [SJJK].[faktura].[2025_fakturaoversikt_juni]
where status = 'ubetalt';