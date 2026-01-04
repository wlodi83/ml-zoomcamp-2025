SELECT *
FROM block
WHERE workspace_id = 2591
  AND (content_json::jsonb ->> 'integration_id') != 'qoxbM4z1vl'
  AND type = 'slack';



select * from integration where workspace_id = 2591;

select * from recipient where workspace_id = 2591;

select * from recipient where id = 19596;

select * from recipient_destination where workspace_id = 2591
and endpoint = 'UFGKR6673';
select * from integration where uid = 'qoxbM4z1vl';

C01SYLK9AES

select * from workspace where organization_id = '';


select * from ab_user where email = 'daria.savvateeva@factris.com';
select * from user_attribute where user_id = 20508;


select * from