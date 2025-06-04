-- TPC-H Query 1

select
        l.returnflag,
        l.linestatus,
        sum(l.quantity) as sum_qty,
        sum(l.extendedprice) as sum_base_price,
        sum(l.extendedprice * (1 - l.discount)) as sum_disc_price,
        sum(l.extendedprice * (1 - l.discount) * (1 + l.tax)) as sum_charge,
        avg(l.quantity) as avg_qty,
        avg(l.extendedprice) as avg_price,
        avg(l.discount) as avg_disc,
        count(*) as count_order
from
        lineitem l
where
        l.shipdate <= date '1998-12-01' - interval '90' day
group by
        l.returnflag,
        l.linestatus
order by
        l.returnflag,
        l.linestatus



-- index_merge=off
-- index_merge_union=on
-- index_merge_sort_union=on
-- index_merge_intersection=on
-- engine_condition_pushdown=on
-- index_condition_pushdown=on
-- mrr=on
-- mrr_cost_based=on
-- block_nested_loop=on
-- batched_key_access=off
-- materialization=off
-- semijoin=on
-- loosescan=on
-- firstmatch=on
-- duplicateweedout=on
-- subquery_materialization_cost_based=on
-- use_index_extensions=on
-- condition_fanout_filter=on
-- derived_merge=on
-- use_invisible_indexes=off
-- skip_scan=on
-- hash_join=off
-- subquery_to_derived=off
-- prefer_ordering_index=on
-- hypergraph_optimizer=off
-- derived_condition_pushdown=on
-- hash_set_operations=on