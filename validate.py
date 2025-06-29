import snowflake.connector
import os

ctx = snowflake.connector.connect(
    user=os.getenv('SF_USER'),
    password=os.getenv('SF_PWD'),
    account=os.getenv('SF_ACCOUNT'),
)
cs = ctx.cursor()
cs.execute("SELECT current_version()")
print(cs.fetchone()[0])
cs.close()
ctx.close()
