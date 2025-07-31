SELECT  
TABLE_CATALOG,
    TABLE_SCHEMA,
    TABLE_NAME, 
    TABLE_TYPE
FROM 
    INFORMATION_SCHEMA.TABLES
WHERE 
    TABLE_TYPE IN ('BASE TABLE', 'VIEW')
    AND TABLE_NAME LIKE '%sola%'; 

SELECT 
    name AS ProcedureName, 
    type_desc AS ObjectType,
	modify_date as modify_date

FROM 
    sys.objects
WHERE 
    type = 'P'; -- 'P' står for Stored Procedures

