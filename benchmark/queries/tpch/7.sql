-- TPC-H Query 7
select
        supp_nation,
        cust_nation,
        l_year,
        sum(volume) as revenue
from
        (
                select
                        n1.name as supp_nation,
                        n2.name as cust_nation,
                        extract(year from l.shipdate) as l_year,  -- Removed CAST
                        l.extendedprice * (1 - l.discount) as volume
                from
                        supplier s,
                        lineitem l,
                        orders o,
                        customer c,
                        nation n1,
                        nation n2
                where
                        s.suppkey = l.suppkey
                        and o.orderkey = l.orderkey
                        and c.custkey = o.custkey
                        and s.nationkey = n1.nationkey
                        and c.nationkey = n2.nationkey
                        and (
                                (n1.name = 'FRANCE' and n2.name = 'GERMANY')
                                or (n1.name = 'GERMANY' and n2.name = 'FRANCE')
                        )
                        and l.shipdate between date '1995-01-01' and date '1996-12-31'
        ) as shipping
group by
        supp_nation,
        cust_nation,
        l_year
order by
        supp_nation,
        cust_nation,
        l_year