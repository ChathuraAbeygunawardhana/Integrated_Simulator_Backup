# Verification matrix

| sha | top | dataflow | memory | array | scheduler | passed | cycles | note |
|-----|-----|----------|--------|-------|-----------|--------|--------|------|
| `0d1a2e072971` | single_dnn_top | OS | STAMP | 4x4 | - | PASS | 49 |  |
| `339d16836b1e` | single_dnn_top | OS | PAGED | 4x4 | - | PASS | 49 |  |
| `72f9ee514ba8` | single_dnn_top | IS | STAMP | 4x4 | - | PASS | 57 |  |
| `5afdde950c69` | single_dnn_top | IS | PAGED | 4x4 | - | PASS | 57 |  |
| `3d0a010bc3cb` | single_dnn_top | WS | STAMP | 4x4 | - | PASS | 78 |  |
| `9f7febddc4ad` | single_dnn_top | WS | PAGED | 4x4 | - | PASS | 78 |  |
| `37de9640ddff` | single_dnn_top | OS | STAMP | 8x8 | - | PASS | 57 |  |
| `7a082aca78e8` | multi_dnn_top | OS | STAMP | 4x4 | FIFO | PASS | 49 |  |
| `3da307347ea7` | multi_dnn_top | OS | STAMP | 4x4 | SJF | PASS | 49 |  |
| `b486a701845d` | multi_dnn_top | OS | STAMP | 4x4 | AIMT | PASS | 49 |  |
| `de0eee9c29e8` | multi_dnn_top | OS | STAMP | 4x4 | BATCHDNN | PASS | 49 |  |
| `349d1d1f0246` | multi_dnn_top | OS | STAMP | 4x4 | BATCHDNN_PP | PASS | 49 |  |
Passed: **12/12**
