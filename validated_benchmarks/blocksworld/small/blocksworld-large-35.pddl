(define (problem blocksworld-large-35)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b3) (on b1 b4) (on b2 b1) (on b4 b0) (on b5 b3) (on b6 b7) (on b7 b8) (on b8 b9) (on b9 b5) (clear b2) (clear b6) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b9) (arm-empty))
  )
)

python main.py generate --num-problems 100 --difficulty large --num-blocks 11 --timeout 100
