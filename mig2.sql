SELECT  
TABLE_CATALOG,
    TABLE_SCHEMA,
    TABLE_NAME, 
    TABLE_TYPE
FROM 
    INFORMATION_SCHEMA.TABLES
WHERE 
    TABLE_TYPE IN ('BASE TABLE', 'VIEW')
    AND TABLE_NAME LIKE '%GEM%'; 

	------------------------------------------------

	DECLARE @SchemaName NVARCHAR(128) = 'dw-pre-prod'; -- Angi skjemanavn her

SELECT   
    TABLE_CATALOG,
    TABLE_SCHEMA,
    TABLE_NAME, 
    TABLE_TYPE
FROM 
    INFORMATION_SCHEMA.TABLES
WHERE 
    TABLE_TYPE IN ('BASE TABLE', 'VIEW')
    AND TABLE_NAME LIKE '%MPC_MpcTripSummary%'
    AND TABLE_SCHEMA = @SchemaName;



----------------------------------------------------
SELECT 
    o.name AS ProcedureName, 
    o.type_desc AS ObjectType,
    CONVERT(VARCHAR, o.modify_date, 23) AS modify_date, -- Datoformat: yyyy-MM-dd
    LEN(m.definition) AS ProcedureSize -- Størrelse på prosedyren i antall tegn
FROM 
    sys.objects o
INNER JOIN 
    sys.sql_modules m ON o.object_id = m.object_id
WHERE 
    o.type = 'P'; -- 'P' står for Stored Procedures


----------------------------------------------
SELECT 
    o.name AS ProcedureName, 
    o.type_desc AS ObjectType,
    CONVERT(VARCHAR, o.modify_date, 23) AS modify_date, -- Datoformat: yyyy-MM-dd
    LEN(m.definition) AS ProcedureSize -- Størrelse på prosedyren i antall tegn
FROM 
    sys.objects o
INNER JOIN 
    sys.sql_modules m ON o.object_id = m.object_id
WHERE 
    o.type = 'P'; -- 'P' står for Stored Procedures
-----------------------------------------------------------
	---old
	------

	SELECT 
    name AS ProcedureName, 
    type_desc AS ObjectType,
	modify_date as modify_date

FROM 
    sys.objects
WHERE 
    type = 'P'; -- 'P' står for Stored Procedures
