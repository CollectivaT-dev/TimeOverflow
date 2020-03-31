# 2020 Col·lectivaT
#
# TimeOverflow queries as string variables

ACTIVE_USERS = """
SELECT members.id, members.organization_id
FROM movements
INNER JOIN "accounts" ON "movements"."account_id" = "accounts"."id"
INNER JOIN "members" ON "accounts"."accountable_id"= "members"."id"
WHERE movements.created_at>'%s' and movements.created_at<'%s' and accounts.accountable_type='Member'
GROUP BY "members"."id";
"""
