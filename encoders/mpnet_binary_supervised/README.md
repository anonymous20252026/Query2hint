---
language: []
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:8000
- loss:TripletLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: SELECT COUNT(*) FROM title as t, movie_info as mi1, kind_type as
    kt, info_type as it1, info_type as it3, info_type as it4, movie_info_idx as mii1,
    movie_info_idx as mii2, movie_keyword as mk, keyword as k, aka_name as an, name
    as n, info_type as it5, person_info as pi1, cast_info as ci, role_type as rt WHERE
    t.id = mi1.movie_id AND t.id = ci.movie_id AND t.id = mii1.movie_id AND t.id =
    mii2.movie_id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND mi1.info_type_id
    = it1.id AND mii1.info_type_id = it3.id AND mii2.info_type_id = it4.id AND t.kind_id
    = kt.id AND (kt.kind IN ('episode','movie')) AND (t.production_year <= 2015) AND
    (t.production_year >= 1975) AND (mi1.info IN ('$10,000','China','Finland','Greece','Ireland','Israel','Paris,
    France','Poland','Sweden','Switzerland','West Germany')) AND (it1.id IN ('105','18','8'))
    AND it3.id = '100' AND it4.id = '101' AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$'
    AND mii2.info::float <= 8.0) AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND
    0.0 <= mii2.info::float) AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 0.0
    <= mii1.info::float) AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mii1.info::float
    <= 10000.0) AND n.id = ci.person_id AND ci.person_id = pi1.person_id AND it5.id
    = pi1.info_type_id AND n.id = pi1.person_id AND n.id = an.person_id AND rt.id
    = ci.role_id AND (n.gender in ('m')) AND (n.name_pcode_nf in ('D1313','D5431','D5456','G4163','H526','L6523','P1426','R2532','S2153','S3163','S5242'))
    AND (ci.note in ('(voice)') OR ci.note IS NULL) AND (rt.role in ('actor')) AND
    (it5.id in ('22'));
  sentences:
  - SELECT mi1.info, pi.info, COUNT(*) FROM title as t, kind_type as kt, movie_info
    as mi1, info_type as it1, cast_info as ci, role_type as rt, name as n, info_type
    as it2, person_info as pi WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND
    mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.movie_id
    = mi1.movie_id AND ci.role_id = rt.id AND n.id = pi.person_id AND pi.info_type_id
    = it2.id AND (it1.id IN ('106')) AND (it2.id IN ('28')) AND (mi1.info ILIKE '%sc%')
    AND (pi.info ILIKE '%ki%') AND (kt.kind IN ('movie','tv mini series','tv movie','tv
    series','video game','video movie')) AND (rt.role IN ('actor','cinematographer','composer','director','editor','guest','miscellaneous
    crew','producer','production designer')) GROUP BY mi1.info, pi.info;
  - SELECT mi1.info, n.name, COUNT(*) FROM title as t, kind_type as kt, movie_info
    as mi1, info_type as it1, cast_info as ci, role_type as rt, name as n, info_type
    as it2, person_info as pi WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND
    mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.movie_id
    = mi1.movie_id AND ci.role_id = rt.id AND n.id = pi.person_id AND pi.info_type_id
    = it2.id AND (it1.id IN ('3')) AND (it2.id IN ('20')) AND (mi1.info IN ('Biography','Crime','History','Music','Musical','Mystery','Romance','Sport'))
    AND (n.name ILIKE '%hel%') AND (kt.kind IN ('video game','video movie')) AND (rt.role
    IN ('cinematographer','composer','editor','writer')) AND (t.production_year <=
    2015) AND (t.production_year >= 1975) GROUP BY mi1.info, n.name;
  - SELECT MIN(mi.info) AS release_date, MIN(miidx.info) AS rating, MIN(t.title) AS
    german_movie FROM company_name AS cn, company_type AS ct, info_type AS it, info_type
    AS it2, kind_type AS kt, movie_companies AS mc, movie_info AS mi, movie_info_idx
    AS miidx, title AS t WHERE cn.country_code ='[de]' AND ct.kind ='production companies'
    AND it.info ='rating' AND it2.info ='release dates' AND kt.kind ='movie' AND mi.movie_id
    = t.id AND it2.id = mi.info_type_id AND kt.id = t.kind_id AND mc.movie_id = t.id
    AND cn.id = mc.company_id AND ct.id = mc.company_type_id AND miidx.movie_id =
    t.id AND it.id = miidx.info_type_id AND mi.movie_id = miidx.movie_id AND mi.movie_id
    = mc.movie_id AND miidx.movie_id = mc.movie_id;
- source_sentence: SELECT n.gender, rt.role, cn.name, COUNT(*) FROM title as t, movie_companies
    as mc, company_name as cn, company_type as ct, kind_type as kt, cast_info as ci,
    name as n, role_type as rt, movie_info as mi1, info_type as it WHERE t.id = mc.movie_id
    AND t.id = ci.movie_id AND t.id = mi1.movie_id AND mi1.movie_id = ci.movie_id
    AND ci.movie_id = mc.movie_id AND cn.id = mc.company_id AND ct.id = mc.company_type_id
    AND kt.id = t.kind_id AND ci.person_id = n.id AND ci.role_id = rt.id AND mi1.info_type_id
    = it.id AND (kt.kind ILIKE '%m%') AND (rt.role IN ('actor','actress','cinematographer','composer','costume
    designer','guest','miscellaneous crew','producer','production designer','writer'))
    AND (t.production_year <= 2015) AND (t.production_year >= 1990) AND (it.id IN
    ('6')) AND (mi1.info ILIKE '%dig%') AND (cn.name ILIKE '%an%') GROUP BY n.gender,
    rt.role, cn.name ORDER BY COUNT(*) DESC;
  sentences:
  - SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as it1, movie_info
    as mi1, cast_info as ci, role_type as rt, name as n, movie_keyword as mk, keyword
    as k, movie_companies as mc, company_type as ct, company_name as cn WHERE t.id
    = ci.movie_id AND t.id = mc.movie_id AND t.id = mi1.movie_id AND t.id = mk.movie_id
    AND mc.company_type_id = ct.id AND mc.company_id = cn.id AND k.id = mk.keyword_id
    AND mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id = n.id AND
    ci.role_id = rt.id AND (it1.id IN ('8')) AND (mi1.info in ('USA')) AND (kt.kind
    in ('episode')) AND (rt.role in ('actress','miscellaneous crew')) AND (n.gender
    in ('f')) AND (n.name_pcode_cf in ('G5242','L2654','U1562')) AND (t.production_year
    <= 2015) AND (t.production_year >= 1975) AND (cn.name in ('ABS-CBN','Granada Television','Sony
    Pictures Home Entertainment','Warner Bros. Television')) AND (ct.kind in ('distributors','production
    companies'));
  - SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as it1, movie_info
    as mi1, cast_info as ci, role_type as rt, name as n, movie_keyword as mk, keyword
    as k, movie_companies as mc, company_type as ct, company_name as cn WHERE t.id
    = ci.movie_id AND t.id = mc.movie_id AND t.id = mi1.movie_id AND t.id = mk.movie_id
    AND mc.company_type_id = ct.id AND mc.company_id = cn.id AND k.id = mk.keyword_id
    AND mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id = n.id AND
    ci.role_id = rt.id AND (it1.id IN ('4')) AND (mi1.info in ('English')) AND (kt.kind
    in ('episode','movie')) AND (rt.role in ('actress')) AND (n.gender in ('f')) AND
    (n.name_pcode_nf in ('A4252','B6161','B6162','J5162','K3451','K3652','L5326'))
    AND (t.production_year <= 2015) AND (t.production_year >= 1990) AND (cn.name in
    ('British Broadcasting Corporation (BBC)','Columbia Broadcasting System (CBS)','National
    Broadcasting Company (NBC)','Warner Home Video')) AND (ct.kind in ('distributors'));
  - SELECT COUNT(*) FROM title as t, kind_type as kt, movie_info as mi1, info_type
    as it1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt,
    name as n WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND t.id = mi2.movie_id
    AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id
    = it2.id AND it1.id = '3' AND it2.id = '4' AND t.kind_id = kt.id AND ci.person_id
    = n.id AND ci.role_id = rt.id AND mi1.info IN ('Crime','Documentary','Drama','History','Musical')
    AND mi2.info IN ('Dutch','English','Finnish','German','Korean','Spanish','Tagalog')
    AND kt.kind IN ('episode','movie','tv movie') AND rt.role IN ('cinematographer','composer')
    AND n.gender IN ('m') AND t.production_year <= 1975 AND 1875 < t.production_year;
- source_sentence: SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as
    it1, movie_info as mi1, cast_info as ci, role_type as rt, name as n, movie_keyword
    as mk, keyword as k, movie_companies as mc, company_type as ct, company_name as
    cn WHERE t.id = ci.movie_id AND t.id = mc.movie_id AND t.id = mi1.movie_id AND
    t.id = mk.movie_id AND mc.company_type_id = ct.id AND mc.company_id = cn.id AND
    k.id = mk.keyword_id AND mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id
    = n.id AND ci.role_id = rt.id AND (it1.id IN ('2')) AND (mi1.info in ('Color'))
    AND (kt.kind in ('episode')) AND (rt.role in ('actress','miscellaneous crew'))
    AND (n.gender in ('f')) AND (n.surname_pcode in ('A52','S532','U15')) AND (t.production_year
    <= 2015) AND (t.production_year >= 1975) AND (cn.name in ('ABS-CBN','Fox Network','Warner
    Bros. Television')) AND (ct.kind in ('distributors','production companies'));
  sentences:
  - SELECT COUNT(*) FROM title as t, movie_info as mi1, kind_type as kt, info_type
    as it1, info_type as it3, info_type as it4, movie_info_idx as mii1, movie_info_idx
    as mii2, movie_keyword as mk, keyword as k, aka_name as an, name as n, info_type
    as it5, person_info as pi1, cast_info as ci, role_type as rt WHERE t.id = mi1.movie_id
    AND t.id = ci.movie_id AND t.id = mii1.movie_id AND t.id = mii2.movie_id AND t.id
    = mk.movie_id AND mk.keyword_id = k.id AND mi1.info_type_id = it1.id AND mii1.info_type_id
    = it3.id AND mii2.info_type_id = it4.id AND t.kind_id = kt.id AND (kt.kind IN
    ('episode','movie')) AND (t.production_year <= 1990) AND (t.production_year >=
    1950) AND (mi1.info IN ('90','USA:30')) AND (it1.id IN ('1')) AND it3.id = '100'
    AND it4.id = '101' AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mii2.info::float
    <= 8.0) AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 0.0 <= mii2.info::float)
    AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 1000.0 <= mii1.info::float)
    AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mii1.info::float <= 10000.0)
    AND n.id = ci.person_id AND ci.person_id = pi1.person_id AND it5.id = pi1.info_type_id
    AND n.id = pi1.person_id AND n.id = an.person_id AND rt.id = ci.role_id AND (n.gender
    in ('m')) AND (n.name_pcode_nf in ('A4152','D1324','D5432','G6252','J5163','J5245','M2424','M2453','P3635','P3652','S3153'))
    AND (ci.note IS NULL) AND (rt.role in ('actor')) AND (it5.id in ('37'));
  - 'SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as it1, movie_info
    as mi1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt,
    name as n, movie_keyword as mk, keyword as k WHERE t.id = ci.movie_id AND t.id
    = mi1.movie_id AND t.id = mi2.movie_id AND t.id = mk.movie_id AND k.id = mk.keyword_id
    AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id
    = it2.id AND (it1.id in (''7'')) AND (it2.id in (''3'')) AND t.kind_id = kt.id
    AND ci.person_id = n.id AND ci.role_id = rt.id AND (mi1.info in (''MET:300 m'',''MET:600
    m'',''OFM:35 mm'',''PCS:Spherical'',''PFM:35 mm'',''RAT:1.33 : 1'',''RAT:1.37
    : 1'')) AND (mi2.info in (''Animation'',''Comedy'',''Crime'',''Drama'',''Family'',''Short'',''Thriller''))
    AND (kt.kind in (''movie'',''tv movie'')) AND (rt.role in (''cinematographer'',''costume
    designer'')) AND (n.gender in (''f'',''m'')) AND (t.production_year <= 1975) AND
    (t.production_year >= 1925);'
  - SELECT MIN(n.name) AS member_in_charnamed_movie, MIN(n.name) AS a1 FROM cast_info
    AS ci, company_name AS cn, keyword AS k, movie_companies AS mc, movie_keyword
    AS mk, name AS n, title AS t WHERE k.keyword ='character-name-in-title' AND n.name  LIKE
    'Z%' AND n.id = ci.person_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND
    mk.keyword_id = k.id AND t.id = mc.movie_id AND mc.company_id = cn.id AND ci.movie_id
    = mc.movie_id AND ci.movie_id = mk.movie_id AND mc.movie_id = mk.movie_id;
- source_sentence: 'SELECT MIN(n.name) AS voicing_actress, MIN(t.title) AS jap_engl_voiced_movie
    FROM aka_name AS an, char_name AS chn, cast_info AS ci, company_name AS cn, info_type
    AS it, movie_companies AS mc, movie_info AS mi, name AS n, role_type AS rt, title
    AS t WHERE ci.note  in (''(voice)'', ''(voice: Japanese version)'', ''(voice)
    (uncredited)'', ''(voice: English version)'') AND cn.country_code =''[us]'' AND
    it.info  = ''release dates'' AND n.gender =''f'' AND rt.role =''actress'' AND
    t.production_year  > 2000 AND t.id = mi.movie_id AND t.id = mc.movie_id AND t.id
    = ci.movie_id AND mc.movie_id = ci.movie_id AND mc.movie_id = mi.movie_id AND
    mi.movie_id = ci.movie_id AND cn.id = mc.company_id AND it.id = mi.info_type_id
    AND n.id = ci.person_id AND rt.id = ci.role_id AND n.id = an.person_id AND ci.person_id
    = an.person_id AND chn.id = ci.person_role_id;'
  sentences:
  - SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as it1, movie_info
    as mi1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt,
    name as n WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND t.id = mi2.movie_id
    AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id
    = it2.id AND (it1.id in ('5')) AND (it2.id in ('3')) AND t.kind_id = kt.id AND
    ci.person_id = n.id AND ci.role_id = rt.id AND (mi1.info IN ('Australia:MA','Belgium:KT','Canada:14','Canada:18','Finland:K-18','Ireland:15A','Netherlands:MG6','Portugal:M/16','Singapore:PG13','South
    Africa:PG','South Korea:12','Spain:13','UK:AA','UK:E','USA:X')) AND (mi2.info
    IN ('Action','Adult','Animation','Biography','Drama','Game-Show','Music','Musical','Mystery','Reality-TV','Sci-Fi','Thriller','War'))
    AND (kt.kind in ('movie','tv movie','tv series','video game','video movie')) AND
    (rt.role in ('costume designer','editor','production designer','writer')) AND
    (n.gender IN ('f')) AND (t.production_year <= 2015) AND (t.production_year >=
    1925);
  - 'SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as it1, movie_info
    as mi1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt,
    name as n, movie_keyword as mk, keyword as k WHERE t.id = ci.movie_id AND t.id
    = mi1.movie_id AND t.id = mi2.movie_id AND t.id = mk.movie_id AND k.id = mk.keyword_id
    AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id
    = it2.id AND (it1.id in (''4'')) AND (it2.id in (''7'')) AND t.kind_id = kt.id
    AND ci.person_id = n.id AND ci.role_id = rt.id AND (mi1.info in (''Czech'',''English'',''Hungarian'',''Italian'',''Japanese'',''Mandarin'',''Portuguese'',''Russian'',''Swedish''))
    AND (mi2.info in (''OFM:35 mm'',''PCS:Kinescope'',''PCS:Spherical'',''PFM:35 mm'',''RAT:1.20
    : 1'',''RAT:1.37 : 1'',''RAT:1.66 : 1'',''RAT:2.35 : 1'')) AND (kt.kind in (''movie'',''tv
    movie'')) AND (rt.role in (''miscellaneous crew'',''writer'')) AND (n.gender in
    (''f'') OR n.gender IS NULL) AND (t.production_year <= 1975) AND (t.production_year
    >= 1925);'
  - 'SELECT COUNT(*) FROM title as t, kind_type as kt, movie_info as mi1, info_type
    as it1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt,
    name as n WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND t.id = mi2.movie_id
    AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id
    = it2.id AND it1.id = ''3'' AND it2.id = ''7'' AND t.kind_id = kt.id AND ci.person_id
    = n.id AND ci.role_id = rt.id AND mi1.info IN (''Action'',''Animation'',''Crime'',''Drama'',''Fantasy'',''Horror'',''Thriller'')
    AND mi2.info IN (''LAB:FotoKem Laboratory, Burbank (CA), USA'',''PCS:Digital Intermediate'',''PCS:Spherical'',''PFM:35
    mm'',''RAT:1.33 : 1'',''RAT:1.85 : 1'') AND kt.kind IN (''episode'',''video movie'')
    AND rt.role IN (''composer'',''miscellaneous crew'') AND n.gender IN (''m'') AND
    t.production_year <= 2010 AND 2000 < t.production_year;'
- source_sentence: SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as
    it1, movie_info as mi1, movie_info as mi2, info_type as it2, cast_info as ci,
    role_type as rt, name as n, movie_keyword as mk, keyword as k WHERE t.id = ci.movie_id
    AND t.id = mi1.movie_id AND t.id = mi2.movie_id AND t.id = mk.movie_id AND k.id
    = mk.keyword_id AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id
    AND mi2.info_type_id = it2.id AND (it1.id in ('8')) AND (it2.id in ('1')) AND
    t.kind_id = kt.id AND ci.person_id = n.id AND ci.role_id = rt.id AND (mi1.info
    in ('France','UK','USA')) AND (mi2.info in ('100','25','30','60','75','90','UK:30','USA:30','USA:60'))
    AND (kt.kind in ('tv series','video game','video movie')) AND (rt.role in ('composer','miscellaneous
    crew')) AND (n.gender in ('f') OR n.gender IS NULL) AND (t.production_year <=
    1990) AND (t.production_year >= 1950);
  sentences:
  - SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as it1, movie_info
    as mi1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt,
    name as n, movie_keyword as mk, keyword as k WHERE t.id = ci.movie_id AND t.id
    = mi1.movie_id AND t.id = mi2.movie_id AND t.id = mk.movie_id AND k.id = mk.keyword_id
    AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id
    = it2.id AND (it1.id in ('7')) AND (it2.id in ('18')) AND t.kind_id = kt.id AND
    ci.person_id = n.id AND ci.role_id = rt.id AND (mi1.info in ('OFM:35 mm')) AND
    (mi2.info in ('Revue Studios, Hollywood, Los Angeles, California, USA','Universal
    Studios - 100 Universal City Plaza, Universal City, California, USA','Warner Brothers
    Burbank Studios - 4000 Warner Boulevard, Burbank, California, USA')) AND (kt.kind
    in ('episode','movie','tv movie')) AND (rt.role in ('director','miscellaneous
    crew')) AND (n.gender in ('f','m')) AND (t.production_year <= 1975) AND (t.production_year
    >= 1875);
  - SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as it1, movie_info
    as mi1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt,
    name as n, movie_keyword as mk, keyword as k WHERE t.id = ci.movie_id AND t.id
    = mi1.movie_id AND t.id = mi2.movie_id AND t.id = mk.movie_id AND k.id = mk.keyword_id
    AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id
    = it2.id AND (it1.id in ('8')) AND (it2.id in ('2')) AND t.kind_id = kt.id AND
    ci.person_id = n.id AND ci.role_id = rt.id AND (mi1.info in ('Germany','Greece'))
    AND (mi2.info in ('Black and White')) AND (kt.kind in ('tv series','video game','video
    movie')) AND (rt.role in ('composer')) AND (n.gender in ('m')) AND (t.production_year
    <= 1975) AND (t.production_year >= 1875) AND (k.keyword IN ('anal-sex','bare-chested-male','husband-wife-relationship','independent-film','non-fiction','tv-mini-series'));
  - 'SELECT COUNT(*) FROM title as t, movie_info as mi1, kind_type as kt, info_type
    as it1, info_type as it3, info_type as it4, movie_info_idx as mii1, movie_info_idx
    as mii2, movie_keyword as mk, keyword as k WHERE t.id = mi1.movie_id AND t.id
    = mii1.movie_id AND t.id = mii2.movie_id AND t.id = mk.movie_id AND mii2.movie_id
    = mii1.movie_id AND mi1.movie_id = mii1.movie_id AND mk.movie_id = mi1.movie_id
    AND mk.keyword_id = k.id AND mi1.info_type_id = it1.id AND mii1.info_type_id =
    it3.id AND mii2.info_type_id = it4.id AND t.kind_id = kt.id AND (kt.kind IN (''episode'',''movie'',''video
    movie'')) AND (t.production_year <= 2015) AND (t.production_year >= 1975) AND
    (mi1.info IN (''English'',''French'',''German'',''Japanese'',''Los Angeles, California,
    USA'',''OFM:35 mm'',''OFM:Video'',''PFM:35 mm'',''RAT:1.33 : 1'',''RAT:1.78 :
    1'',''RAT:1.85 : 1'',''RAT:16:9 HD'',''RAT:2.35 : 1'',''Spanish'')) AND (it1.id
    IN (''18'',''4'',''7'')) AND it3.id = ''100'' AND it4.id = ''101'' AND (mii2.info
    ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$'' AND mii2.info::float <= 7.0) AND (mii2.info
    ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$'' AND 3.0 <= mii2.info::float) AND (mii1.info
    ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$'' AND 800.0 <= mii1.info::float) AND (mii1.info
    ~ ''^(?:[1-9]\d*|0)?(?:\.\d+)?$'' AND mii1.info::float <= 31000.0);'
datasets: []
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision e8c3b32edf5434bc2275fc9bab85f82640a19130 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: MPNetModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    "SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as it1, movie_info as mi1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt, name as n, movie_keyword as mk, keyword as k WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND t.id = mi2.movie_id AND t.id = mk.movie_id AND k.id = mk.keyword_id AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id = it2.id AND (it1.id in ('8')) AND (it2.id in ('1')) AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.role_id = rt.id AND (mi1.info in ('France','UK','USA')) AND (mi2.info in ('100','25','30','60','75','90','UK:30','USA:30','USA:60')) AND (kt.kind in ('tv series','video game','video movie')) AND (rt.role in ('composer','miscellaneous crew')) AND (n.gender in ('f') OR n.gender IS NULL) AND (t.production_year <= 1990) AND (t.production_year >= 1950);",
    "SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as it1, movie_info as mi1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt, name as n, movie_keyword as mk, keyword as k WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND t.id = mi2.movie_id AND t.id = mk.movie_id AND k.id = mk.keyword_id AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id = it2.id AND (it1.id in ('8')) AND (it2.id in ('2')) AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.role_id = rt.id AND (mi1.info in ('Germany','Greece')) AND (mi2.info in ('Black and White')) AND (kt.kind in ('tv series','video game','video movie')) AND (rt.role in ('composer')) AND (n.gender in ('m')) AND (t.production_year <= 1975) AND (t.production_year >= 1875) AND (k.keyword IN ('anal-sex','bare-chested-male','husband-wife-relationship','independent-film','non-fiction','tv-mini-series'));",
    "SELECT COUNT(*) FROM title as t, movie_info as mi1, kind_type as kt, info_type as it1, info_type as it3, info_type as it4, movie_info_idx as mii1, movie_info_idx as mii2, movie_keyword as mk, keyword as k WHERE t.id = mi1.movie_id AND t.id = mii1.movie_id AND t.id = mii2.movie_id AND t.id = mk.movie_id AND mii2.movie_id = mii1.movie_id AND mi1.movie_id = mii1.movie_id AND mk.movie_id = mi1.movie_id AND mk.keyword_id = k.id AND mi1.info_type_id = it1.id AND mii1.info_type_id = it3.id AND mii2.info_type_id = it4.id AND t.kind_id = kt.id AND (kt.kind IN ('episode','movie','video movie')) AND (t.production_year <= 2015) AND (t.production_year >= 1975) AND (mi1.info IN ('English','French','German','Japanese','Los Angeles, California, USA','OFM:35 mm','OFM:Video','PFM:35 mm','RAT:1.33 : 1','RAT:1.78 : 1','RAT:1.85 : 1','RAT:16:9 HD','RAT:2.35 : 1','Spanish')) AND (it1.id IN ('18','4','7')) AND it3.id = '100' AND it4.id = '101' AND (mii2.info ~ '^(?:[1-9]\\d*|0)?(?:\\.\\d+)?$' AND mii2.info::float <= 7.0) AND (mii2.info ~ '^(?:[1-9]\\d*|0)?(?:\\.\\d+)?$' AND 3.0 <= mii2.info::float) AND (mii1.info ~ '^(?:[1-9]\\d*|0)?(?:\\.\\d+)?$' AND 800.0 <= mii1.info::float) AND (mii1.info ~ '^(?:[1-9]\\d*|0)?(?:\\.\\d+)?$' AND mii1.info::float <= 31000.0);",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 8,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                            | sentence_1                                                                            | sentence_2                                                                            |
  |:--------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                                | string                                                                                | string                                                                                |
  | details | <ul><li>min: 122 tokens</li><li>mean: 400.56 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 151 tokens</li><li>mean: 401.57 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 122 tokens</li><li>mean: 397.67 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | sentence_2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>SELECT COUNT(*) FROM title as t, movie_info as mi1, kind_type as kt, info_type as it1, info_type as it3, info_type as it4, movie_info_idx as mii1, movie_info_idx as mii2, movie_keyword as mk, keyword as k, aka_name as an, name as n, info_type as it5, person_info as pi1, cast_info as ci, role_type as rt WHERE t.id = mi1.movie_id AND t.id = ci.movie_id AND t.id = mii1.movie_id AND t.id = mii2.movie_id AND t.id = mk.movie_id AND mk.keyword_id = k.id AND mi1.info_type_id = it1.id AND mii1.info_type_id = it3.id AND mii2.info_type_id = it4.id AND t.kind_id = kt.id AND (kt.kind IN ('tv movie','video movie')) AND (t.production_year <= 2015) AND (t.production_year >= 1975) AND (mi1.info IN ('Los Angeles, California, USA','OFM:35 mm')) AND (it1.id IN ('107','18','7')) AND it3.id = '100' AND it4.id = '101' AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mii2.info::float <= 11.0) AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 7.0 <= mii2.info::float) AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 5000.0 <= mii1.info::float) AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mii1.info::float <= 500000.0) AND n.id = ci.person_id AND ci.person_id = pi1.person_id AND it5.id = pi1.info_type_id AND n.id = pi1.person_id AND n.id = an.person_id AND rt.id = ci.role_id AND (n.gender in ('m')) AND (n.name_pcode_nf in ('D1316','F6521','J5236','J5252','M2425','P3625','S3151')) AND (ci.note IS NULL) AND (rt.role in ('actor')) AND (it5.id in ('32'));</code>                                                                                                                                      | <code>SELECT mi1.info, pi.info, COUNT(*) FROM title as t, kind_type as kt, movie_info as mi1, info_type as it1, cast_info as ci, role_type as rt, name as n, info_type as it2, person_info as pi WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.movie_id = mi1.movie_id AND ci.role_id = rt.id AND n.id = pi.person_id AND pi.info_type_id = it2.id AND (it1.id IN ('16')) AND (it2.id IN ('21')) AND (mi1.info ILIKE '%germ%') AND (pi.info ILIKE '%19%') AND (kt.kind IN ('episode','movie','tv mini series','tv movie','video movie')) AND (rt.role IN ('cinematographer','composer','costume designer')) GROUP BY mi1.info, pi.info;</code>                                                                                                                                                                                                                                                                            | <code>SELECT MIN(mi_idx.info) AS rating, MIN(t.title) AS northern_dark_movie FROM info_type AS it1, info_type AS it2, keyword AS k, kind_type AS kt, movie_info AS mi, movie_info_idx AS mi_idx, movie_keyword AS mk, title AS t WHERE it1.info  = 'countries' AND it2.info  = 'rating' AND k.keyword  in ('murder', 'murder-in-title', 'blood', 'violence') AND kt.kind  = 'movie' AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'USA', 'American') AND mi_idx.info  < '8.5' AND t.production_year  > 2010 AND kt.id = t.kind_id AND t.id = mi.movie_id AND t.id = mk.movie_id AND t.id = mi_idx.movie_id AND mk.movie_id = mi.movie_id AND mk.movie_id = mi_idx.movie_id AND mi.movie_id = mi_idx.movie_id AND k.id = mk.keyword_id AND it1.id = mi.info_type_id AND it2.id = mi_idx.info_type_id;</code>                                                                                                                                                                                                                                                               |
  | <code>SELECT COUNT(*) FROM title as t, kind_type as kt, info_type as it1, movie_info as mi1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt, name as n, movie_keyword as mk, keyword as k WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND t.id = mi2.movie_id AND t.id = mk.movie_id AND k.id = mk.keyword_id AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id = it2.id AND (it1.id in ('6')) AND (it2.id in ('5')) AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.role_id = rt.id AND (mi1.info in ('Mono','Silent')) AND (mi2.info in ('Argentina:16','Canada:G','Finland:(Banned)','Finland:K-12','Iceland:16','India:U','Singapore:PG','Sweden:11','UK:18','USA:PG','USA:Passed','USA:TV-PG')) AND (kt.kind in ('tv series','video game','video movie')) AND (rt.role in ('producer')) AND (n.gender in ('m') OR n.gender IS NULL) AND (t.production_year <= 1975) AND (t.production_year >= 1875);</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | <code>SELECT COUNT(*) FROM title as t, kind_type as kt, movie_info as mi1, info_type as it1, movie_info as mi2, info_type as it2, cast_info as ci, role_type as rt, name as n WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND t.id = mi2.movie_id AND mi1.movie_id = mi2.movie_id AND mi1.info_type_id = it1.id AND mi2.info_type_id = it2.id AND it1.id = '3' AND it2.id = '4' AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.role_id = rt.id AND mi1.info IN ('Action','Biography','Comedy','Crime','Drama','Romance','Short') AND mi2.info IN ('Czech','Danish','English','French','German','Korean','Spanish','Tagalog') AND kt.kind IN ('episode','tv movie') AND rt.role IN ('editor','writer') AND n.gender IN ('f','m') AND t.production_year <= 1975 AND 1875 < t.production_year;</code>                                                                                                                                                                                                   | <code>SELECT n.gender, rt.role, cn.name, COUNT(*) FROM title as t, movie_companies as mc, company_name as cn, company_type as ct, kind_type as kt, cast_info as ci, name as n, role_type as rt, movie_info as mi1, info_type as it1, person_info as pi, info_type as it2 WHERE t.id = mc.movie_id AND t.id = ci.movie_id AND t.id = mi1.movie_id AND mi1.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND cn.id = mc.company_id AND ct.id = mc.company_type_id AND kt.id = t.kind_id AND ci.person_id = n.id AND ci.role_id = rt.id AND mi1.info_type_id = it1.id AND n.id = pi.person_id AND pi.info_type_id = it2.id AND ci.person_id = pi.person_id AND (kt.kind IN ('episode','movie','tv movie','video game','video movie')) AND (rt.role IN ('actor','actress','cinematographer','composer','costume designer','guest','miscellaneous crew','production designer','writer')) AND (t.production_year <= 2015) AND (t.production_year >= 1875) AND (it1.id IN ('8')) AND (mi1.info ILIKE '%l%') AND (pi.info ILIKE '%nov%') AND (it2.id IN ('38')) GROUP BY n.gender, rt.role, cn.name ORDER BY COUNT(*) DESC;</code> |
  | <code>SELECT COUNT(*) FROM title as t, movie_info as mi1, kind_type as kt, info_type as it1, info_type as it3, info_type as it4, movie_info_idx as mii1, movie_info_idx as mii2, aka_name as an, name as n, info_type as it5, person_info as pi1, cast_info as ci, role_type as rt WHERE t.id = mi1.movie_id AND t.id = ci.movie_id AND t.id = mii1.movie_id AND t.id = mii2.movie_id AND mii2.movie_id = mii1.movie_id AND mi1.movie_id = mii1.movie_id AND mi1.info_type_id = it1.id AND mii1.info_type_id = it3.id AND mii2.info_type_id = it4.id AND t.kind_id = kt.id AND (kt.kind IN ('episode')) AND (t.production_year <= 1975) AND (t.production_year >= 1875) AND (mi1.info IN ('Desilu Studios - 9336 W. Washington Blvd., Culver City, California, USA','USA')) AND (it1.id IN ('17','18','8')) AND it3.id = '100' AND it4.id = '101' AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mii2.info::float <= 8.0) AND (mii2.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 0.0 <= mii2.info::float) AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND 0.0 <= mii1.info::float) AND (mii1.info ~ '^(?:[1-9]\d*|0)?(?:\.\d+)?$' AND mii1.info::float <= 1000.0) AND n.id = ci.person_id AND ci.person_id = pi1.person_id AND it5.id = pi1.info_type_id AND n.id = pi1.person_id AND n.id = an.person_id AND ci.person_id = an.person_id AND an.person_id = pi1.person_id AND rt.id = ci.role_id AND (n.gender in ('f','m')) AND (n.name_pcode_nf in ('A4163','B6563','D1316','F6521','F6523','F6524','J3261','J5216','J525','J5262','P3616','R1635','R3565','W4362')) AND (ci.note IS NULL) AND (rt.role in ('actor','actress')) AND (it5.id in ('37'));</code> | <code>SELECT mi1.info, n.name, COUNT(*) FROM title as t, kind_type as kt, movie_info as mi1, info_type as it1, cast_info as ci, role_type as rt, name as n, info_type as it2, person_info as pi WHERE t.id = ci.movie_id AND t.id = mi1.movie_id AND mi1.info_type_id = it1.id AND t.kind_id = kt.id AND ci.person_id = n.id AND ci.movie_id = mi1.movie_id AND ci.role_id = rt.id AND n.id = pi.person_id AND pi.info_type_id = it2.id AND (it1.id IN ('2','3','5')) AND (it2.id IN ('17')) AND (mi1.info IN ('Australia:M','Australia:PG','Black and White','Comedy','Crime','Drama','Fantasy','Music','Mystery','Reality-TV','Sci-Fi','Short','Singapore:PG','Sport','Talk-Show','Thriller','USA:M','USA:R','USA:T','War')) AND (n.name ILIKE '%danie%') AND (kt.kind IN ('tv movie','tv series','video game')) AND (rt.role IN ('actress','costume designer','director','miscellaneous crew','producer')) AND (t.production_year <= 2015) AND (t.production_year >= 1925) GROUP BY mi1.info, n.name;</code> | <code>SELECT t.title, n.name, cn.name, COUNT(*) FROM title as t, movie_keyword as mk, keyword as k, movie_companies as mc, company_name as cn, company_type as ct, kind_type as kt, cast_info as ci, name as n, role_type as rt WHERE t.id = mk.movie_id AND t.id = mc.movie_id AND t.id = ci.movie_id AND ci.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND k.id = mk.keyword_id AND cn.id = mc.company_id AND ct.id = mc.company_type_id AND kt.id = t.kind_id AND ci.person_id = n.id AND ci.role_id = rt.id AND (t.title ILIKE '%no%') AND (n.surname_pcode ILIKE '%t12%') AND (cn.name ILIKE '%broa%') AND (kt.kind IN ('episode','tv movie','video game','video movie')) AND (rt.role IN ('actor','actress','cinematographer','composer','costume designer','director','editor','miscellaneous crew','producer','writer')) GROUP BY t.title, n.name, cn.name ORDER BY COUNT(*) DESC;</code>                                                                                                                                                                                         |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 5
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `disable_tqdm`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: True
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch | Step | Training Loss |
|:-----:|:----:|:-------------:|
| 0.5   | 500  | 4.8705        |
| 1.0   | 1000 | 4.7653        |
| 1.5   | 1500 | 4.5631        |
| 2.0   | 2000 | 4.459         |
| 2.5   | 2500 | 4.4158        |
| 3.0   | 3000 | 4.4031        |


### Framework Versions
- Python: 3.10.19
- Sentence Transformers: 3.0.1
- Transformers: 4.44.2
- PyTorch: 2.6.0+cu124
- Accelerate: 0.33.0
- Datasets: 3.0.1
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification}, 
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->