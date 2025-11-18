(define (problem blocksworld-large-37)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b1) (on-table b13) (on-table b3) (on-table b6) (on b0 b13) (on b10 b6) (on b11 b8) (on b2 b5) (on b4 b1) (on b5 b4) (on b7 b3) (on b8 b0) (on b9 b7) (clear b10) (clear b11) (clear b2) (clear b9) (holding b12)
  )
  (:goal
    (and (on-table b0) (on-table b1) (on-table b2) (on-table b3) (on b10 b6) (on b11 b7) (on b12 b8) (on b13 b9) (on b4 b0) (on b5 b1) (on b6 b2) (on b7 b3) (on b8 b4) (on b9 b5) (clear b10) (clear b11) (clear b12) (clear b13) (arm-empty))
  )
)
