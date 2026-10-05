//! h18 task bank: 36 domain ops (finance / business / science), canonical specs for `neural_mesh::fin`, and a word-problem
//! RENDERER with a SEEN phrase bank (what the extractor is developed against) and an UNSEEN bank (synonyms, other frames).
//! Text levels (g21 style): L0 plain, L1 formatting (money signs, "percent", shuffled order, filler), L2 number words +
//! distractor numbers, L3 typos, L4 degraded (a quantity dropped or a conflicting second value added).
use neural_mesh::fin::fmt;
use neural_mesh::router::Rng;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum K {
    /// text "8.5%", spec 0.085
    Rate,
    /// text "5.5%", spec 5.5 (percent number)
    Pct,
    Money,
    Count,
    /// raw decimal token kept verbatim
    Num,
    List,
    Unit,
    Formula,
    Species,
    Eq,
    E,
}
pub struct En {
    pub val: &'static str,
    pub seen: &'static [&'static str],
    pub unseen: &'static [&'static str],
}
pub struct Slot {
    pub key: &'static str,
    pub kind: K,
    pub opt: bool,
    pub seen: &'static [&'static str],
    pub unseen: &'static [&'static str],
    pub en: &'static [En],
    pub cue: &'static [&'static str],
}
pub struct OpDef {
    pub op: &'static str,
    /// 0 finance, 1 business/economics, 2 science
    pub dom: usize,
    /// routing sub-domain 0..5
    pub sub: usize,
    pub slots: &'static [Slot],
    pub ask_seen: &'static [&'static str],
    pub ask_unseen: &'static [&'static str],
}

macro_rules! sl {
    ($k:expr, $kind:expr, [$($s:expr),*], [$($u:expr),*]) => {
        Slot { key: $k, kind: $kind, opt: false, seen: &[$($s),*], unseen: &[$($u),*], en: &[], cue: &[] }
    };
}
macro_rules! sli {
    ($k:expr, $kind:expr, [$($c:expr),*]) => {
        Slot { key: $k, kind: $kind, opt: false, seen: &[], unseen: &[], en: &[], cue: &[$($c),*] }
    };
}
macro_rules! slo {
    ($k:expr, $kind:expr, [$($s:expr),*], [$($u:expr),*]) => {
        Slot { key: $k, kind: $kind, opt: true, seen: &[$($s),*], unseen: &[$($u),*], en: &[], cue: &[] }
    };
}
const fn en(val: &'static str, seen: &'static [&'static str], unseen: &'static [&'static str]) -> En {
    En { val, seen, unseen }
}

const EN_FREQ: &[En] = &[
    en("1", &["annually"], &["yearly", "once a year"]),
    en("2", &["semiannually", "semi-annually"], &["twice a year", "every six months"]),
    en("4", &["quarterly"], &["every quarter", "every three months"]),
    en("12", &["monthly"], &["every month", "each month"]),
];
const EN_FREQ_BOND: &[En] = &[en("1", &["annually"], &["once a year"]), en("2", &["semiannually"], &["every six months"])];
const EN_KIND: &[En] = &[en("call", &["call"], &["buy"]), en("put", &["put"], &["sell"])];
const EN_POS: &[En] = &[en("long", &["long"], &["bought", "purchased"]), en("short", &["short"], &["sold", "written"])];
const EN_ERR: &[En] = &[en("add", &["sum"], &["total"]), en("mul", &["product"], &["multiplication"])];
const EN_GAS: &[En] = &[
    en("p", &["pressure"], &["pressure"]),
    en("v", &["volume"], &["capacity"]),
    en("n", &["number of moles", "amount of gas"], &["quantity of gas", "amount of substance"]),
    en("t", &["temperature"], &["absolute temperature"]),
];
const EN_KIN: &[En] = &[
    en("u", &["initial velocity"], &["starting speed"]),
    en("v", &["final velocity"], &["end speed"]),
    en("a", &["acceleration"], &["rate of change of velocity"]),
    en("t", &["time"], &["duration"]),
    en("s", &["distance"], &["displacement"]),
];

pub const LEADS_SEEN: [&[&str]; 3] = [
    &["A company is evaluating an investment.", "An analyst is reviewing a financing problem."],
    &["A small business owner is reviewing the books.", "A manager is preparing a report."],
    &["A student is working on a lab problem.", "An engineer is checking a calculation."],
];
pub const LEADS_UNSEEN: [&[&str]; 3] = [
    &["The treasury team of a mid-sized firm has a question.", "A client of an advisory desk wants a figure."],
    &["The finance lead at a retailer needs a quick number.", "Management is looking at last quarter's numbers."],
    &["During a practical session someone needs a figure.", "A technician is double checking a number."],
];

pub static OPS: &[OpDef] = &[
    OpDef {
        op: "npv",
        dom: 0,
        sub: 0,
        slots: &[
            sl!("rate", K::Rate, ["The discount rate is {v}.", "Use a discount rate of {v}."], ["The required return is {v}.", "Assume a hurdle rate of {v}."]),
            sl!("cf", K::List, ["The cash flows are {v}.", "Year by year the cash flows are {v}."], ["Yearly amounts come to {v}.", "The expected net amounts per year are {v}."]),
        ],
        ask_seen: &["What is the net present value?", "Calculate the NPV."],
        ask_unseen: &["What is the present worth of the venture?", "Is the project worth it in today's money? Give the figure."],
    },
    OpDef {
        op: "irr",
        dom: 0,
        sub: 0,
        slots: &[sl!(
            "cf",
            K::List,
            ["The cash flows are {v}.", "The project's cash flows over the years are {v}."],
            ["The amounts received or paid each year are {v}.", "Net flows by year: {v}."]
        )],
        ask_seen: &["What is the internal rate of return?", "Find the IRR."],
        ask_unseen: &["What yearly return does the venture earn on this stream?", "At what rate does the stream pay back exactly?"],
    },
    OpDef {
        op: "compound",
        dom: 0,
        sub: 1,
        slots: &[
            sl!("p", K::Money, ["The principal is {v}.", "An initial deposit of {v} is made."], ["Someone puts away {v} to begin with.", "The starting balance is {v}."]),
            sl!("r", K::Rate, ["The annual interest rate is {v}.", "It earns {v} interest per year."], ["The nominal yearly yield is {v}.", "The account pays {v} APR."]),
            sl!("n", K::Count, ["The money is invested for {v} years.", "The term is {v} years."], ["It stays there for {v} years in total.", "The horizon is {v} years."]),
            Slot { key: "m", kind: K::E, opt: false, seen: &["Interest is compounded {v}."], unseen: &["Interest is credited {v}."], en: EN_FREQ, cue: &[] },
        ],
        ask_seen: &["What is the future value?", "How much will the account be worth at the end?"],
        ask_unseen: &["What will the balance come to at the end of the term?", "Work out the final amount."],
    },
    OpDef {
        op: "annpv",
        dom: 0,
        sub: 1,
        slots: &[
            sl!("pmt", K::Money, ["The annuity pays {v} at the end of each year.", "A payment of {v} is received each year."], ["Every year brings in {v}.", "Annual instalments of {v} are made."]),
            sl!("r", K::Rate, ["The discount rate is {v}.", "Use an interest rate of {v}."], ["Discount at {v}.", "Take the yield to be {v}."]),
            sl!("n", K::Count, ["The annuity lasts {v} years.", "There are {v} payments."], ["The stream runs for {v} years.", "It goes on for {v} years."]),
        ],
        ask_seen: &["What is the present value of the annuity?"],
        ask_unseen: &["What is the stream worth today?"],
    },
    OpDef {
        op: "loanpmt",
        dom: 0,
        sub: 1,
        slots: &[
            sl!("p", K::Money, ["The loan amount is {v}.", "A loan of {v} is taken out."], ["The amount borrowed is {v}.", "Someone borrows {v}."]),
            sl!("r", K::Rate, ["The interest rate is {v} per month.", "The monthly rate is {v}."], ["Interest accrues at {v} each month.", "Each month the lender charges {v}."]),
            sl!("n", K::Count, ["It is repaid in {v} monthly payments.", "The loan has {v} monthly instalments."], ["The repayment period is {v} months.", "Repayment takes {v} months."]),
        ],
        ask_seen: &["What is the monthly payment?"],
        ask_unseen: &["How much has to be paid each month?"],
    },
    OpDef {
        op: "amort",
        dom: 0,
        sub: 1,
        slots: &[
            sl!("p", K::Money, ["The loan amount is {v}.", "A loan of {v} is taken out."], ["The amount borrowed is {v}.", "Someone borrows {v}."]),
            sl!("r", K::Rate, ["The interest rate is {v} per month.", "The monthly rate is {v}."], ["Interest accrues at {v} each month.", "Each month the lender charges {v}."]),
            sl!("n", K::Count, ["It is repaid in {v} monthly payments.", "The loan has {v} monthly instalments."], ["The repayment period is {v} months.", "Repayment takes {v} months."]),
            sl!("k", K::Count, ["After {v} payments have been made, the loan is checked.", "Look at the loan after payment number {v}."], ["Consider the point right after instalment {v}.", "Stop at instalment {v} and look at the debt."]),
        ],
        ask_seen: &["What is the remaining balance?"],
        ask_unseen: &["How much principal is still owed then?"],
    },
    OpDef {
        op: "bondprice",
        dom: 0,
        sub: 0,
        slots: &[
            sl!("face", K::Money, ["The bond has a face value of {v}.", "Par value is {v}."], ["The bond redeems for {v} at maturity.", "Each bond is redeemed at {v}."]),
            sl!("cpn", K::Rate, ["The coupon rate is {v}.", "It pays a {v} coupon."], ["Interest on the bond is {v} of par.", "The stated rate on the bond is {v}."]),
            sl!("y", K::Rate, ["The yield to maturity is {v}.", "The market yield is {v}."], ["Investors currently demand {v}.", "Comparable bonds return {v}."]),
            sl!("n", K::Count, ["The bond matures in {v} years.", "{v} years remain to maturity."], ["It runs for another {v} years.", "Redemption comes in {v} years."]),
            Slot { key: "f", kind: K::E, opt: false, seen: &["Coupons are paid {v}."], unseen: &["Coupons arrive {v}."], en: EN_FREQ_BOND, cue: &[] },
        ],
        ask_seen: &["What is the price of the bond?"],
        ask_unseen: &["How much should the bond sell for?"],
    },
    OpDef {
        op: "bondyield",
        dom: 0,
        sub: 0,
        slots: &[
            sl!("face", K::Money, ["The bond has a face value of {v}.", "Par value is {v}."], ["The bond redeems for {v} at maturity.", "Each bond is redeemed at {v}."]),
            sl!("cpn", K::Rate, ["The coupon rate is {v}.", "It pays a {v} coupon."], ["Interest on the bond is {v} of par.", "The stated rate on the bond is {v}."]),
            sl!("price", K::Money, ["The bond currently trades at {v}.", "The market price is {v}."], ["It changes hands at {v}.", "Someone paid {v} for it."]),
            sl!("n", K::Count, ["The bond matures in {v} years.", "{v} years remain to maturity."], ["It runs for another {v} years.", "Redemption comes in {v} years."]),
            Slot { key: "f", kind: K::E, opt: false, seen: &["Coupons are paid {v}."], unseen: &["Coupons arrive {v}."], en: EN_FREQ_BOND, cue: &[] },
        ],
        ask_seen: &["What is the yield to maturity?"],
        ask_unseen: &["What annual return does a buyer earn by holding it?"],
    },
    OpDef {
        op: "bondmod",
        dom: 0,
        sub: 0,
        slots: &[
            sl!("face", K::Money, ["The bond has a face value of {v}.", "Par value is {v}."], ["The bond redeems for {v} at maturity.", "Each bond is redeemed at {v}."]),
            sl!("cpn", K::Rate, ["The coupon rate is {v}.", "It pays a {v} coupon."], ["Interest on the bond is {v} of par.", "The stated rate on the bond is {v}."]),
            sl!("y", K::Rate, ["The yield to maturity is {v}.", "The market yield is {v}."], ["Investors currently demand {v}.", "Comparable bonds return {v}."]),
            sl!("n", K::Count, ["The bond matures in {v} years.", "{v} years remain to maturity."], ["It runs for another {v} years.", "Redemption comes in {v} years."]),
            Slot { key: "f", kind: K::E, opt: false, seen: &["Coupons are paid {v}."], unseen: &["Coupons arrive {v}."], en: EN_FREQ_BOND, cue: &[] },
        ],
        ask_seen: &["What is the modified duration?"],
        ask_unseen: &["How sensitive is the bond price to yield changes (duration adjusted for the yield)?"],
    },
    OpDef {
        op: "cagr",
        dom: 0,
        sub: 0,
        slots: &[
            sl!("s", K::Money, ["The starting value is {v}.", "The investment began at {v}."], ["It was worth {v} at the outset.", "Initially the figure stood at {v}."]),
            sl!("e", K::Money, ["The ending value is {v}.", "It grew to {v}."], ["It ended up at {v}.", "The final figure is {v}."]),
            sl!("n", K::Count, ["The period is {v} years.", "This took {v} years."], ["Over {v} years.", "The elapsed time is {v} years."]),
        ],
        ask_seen: &["What is the compound annual growth rate?", "Find the CAGR."],
        ask_unseen: &["What constant yearly rate would produce that growth?", "At what steady pace did it grow each year?"],
    },
    OpDef {
        op: "wacc",
        dom: 0,
        sub: 0,
        slots: &[
            sl!("e", K::Money, ["The market value of equity is {v}.", "Equity is worth {v}."], ["Shareholders' stake is valued at {v}.", "The firm's stock is worth {v} in total."]),
            sl!("d", K::Money, ["The market value of debt is {v}.", "Debt is worth {v}."], ["Borrowings amount to {v}.", "The firm owes {v} in total."]),
            sl!("re", K::Rate, ["The cost of equity is {v}.", "Equity investors require {v}."], ["Stockholders expect {v}.", "The return on the stock is expected to be {v}."]),
            sl!("rd", K::Rate, ["The cost of debt is {v}.", "Debt costs {v} before tax."], ["Lenders charge {v}.", "The interest on borrowings is {v}."]),
            sl!("t", K::Rate, ["The tax rate is {v}.", "The corporate tax rate is {v}."], ["Profits are taxed at {v}.", "The firm pays {v} in taxes."]),
        ],
        ask_seen: &["What is the WACC?", "Calculate the weighted average cost of capital."],
        ask_unseen: &["What blended financing cost does the firm face?", "What is the overall required return across funding sources?"],
    },
    OpDef {
        op: "perp",
        dom: 0,
        sub: 1,
        slots: &[
            sl!("c", K::Money, ["The perpetuity pays {v} per year.", "A payment of {v} is received every year forever."], ["Forever after, {v} arrives each year.", "A never-ending stream of {v} annually."]),
            sl!("r", K::Rate, ["The discount rate is {v}.", "Use an interest rate of {v}."], ["The yield is {v}.", "Discount at {v}."]),
        ],
        ask_seen: &["What is the present value of the perpetuity?"],
        ask_unseen: &["What is that endless stream worth now?"],
    },
    OpDef {
        op: "payoff",
        dom: 0,
        sub: 0,
        slots: &[
            Slot { key: "kind", kind: K::E, opt: false, seen: &["The option is a {v}."], unseen: &["The contract is a right to {v} the stock."], en: EN_KIND, cue: &[] },
            Slot { key: "pos", kind: K::E, opt: false, seen: &["The investor has a {v} position."], unseen: &["The option was {v} by the investor."], en: EN_POS, cue: &[] },
            sl!("s", K::Money, ["The stock price at expiry is {v}.", "At expiration the stock trades at {v}."], ["On the last day the shares are at {v}.", "At the end the underlying finishes at {v}."]),
            sl!("k", K::Money, ["The strike price is {v}.", "The exercise price is {v}."], ["The contract fixes {v} as the price.", "It is struck at {v}."]),
            sl!("prem", K::Money, ["The premium is {v}.", "The option cost {v}."], ["Buying or writing it involved {v}.", "A fee of {v} changed hands."]),
        ],
        ask_seen: &["What is the profit at expiry?"],
        ask_unseen: &["What is the net result of the position?"],
    },
    OpDef {
        op: "payback",
        dom: 1,
        sub: 2,
        slots: &[sl!(
            "cf",
            K::List,
            ["The cash flows are {v}.", "The project costs and returns are {v}."],
            ["Yearly amounts are {v}.", "Each year's flow is {v}."]
        )],
        ask_seen: &["What is the payback period in years?"],
        ask_unseen: &["After how many years is the outlay recovered?"],
    },
    OpDef {
        op: "breakeven",
        dom: 1,
        sub: 2,
        slots: &[
            sl!("fc", K::Money, ["Fixed costs are {v}.", "The fixed costs total {v}."], ["Overheads come to {v}.", "Standing costs of {v} must be covered."]),
            sl!("price", K::Money, ["The selling price per unit is {v}.", "Each unit sells for {v}."], ["Customers pay {v} per item.", "One piece is priced at {v}."]),
            sl!("vc", K::Money, ["The variable cost per unit is {v}.", "Each unit costs {v} to make."], ["Producing one piece takes {v} of materials and labour.", "Direct cost per item is {v}."]),
        ],
        ask_seen: &["How many units must be sold to break even?"],
        ask_unseen: &["How many items need to be sold before profit starts?"],
    },
    OpDef {
        op: "current",
        dom: 1,
        sub: 2,
        slots: &[
            sl!("ca", K::Money, ["Current assets are {v}.", "The firm has current assets of {v}."], ["Short-term assets total {v}.", "Liquid holdings amount to {v}."]),
            sl!("cl", K::Money, ["Current liabilities are {v}.", "It owes {v} in current liabilities."], ["Short-term obligations are {v}.", "Bills due soon come to {v}."]),
        ],
        ask_seen: &["What is the current ratio?"],
        ask_unseen: &["How well can short-term bills be covered?"],
    },
    OpDef {
        op: "roe",
        dom: 1,
        sub: 2,
        slots: &[
            sl!("ni", K::Money, ["Net income is {v}.", "The company earned a net income of {v}."], ["Profit after everything came to {v}.", "Bottom-line earnings were {v}."]),
            sl!("eq", K::Money, ["Shareholders' equity is {v}.", "Equity is {v}."], ["Owners' funds stand at {v}.", "Book value of the stake is {v}."]),
        ],
        ask_seen: &["What is the return on equity?", "Compute the ROE."],
        ask_unseen: &["How much does the firm earn per unit of owners' money?"],
    },
    OpDef {
        op: "gmargin",
        dom: 1,
        sub: 2,
        slots: &[
            sl!("rev", K::Money, ["Revenue is {v}.", "Sales were {v}."], ["Turnover came to {v}.", "Total takings were {v}."]),
            sl!("cogs", K::Money, ["The cost of goods sold is {v}.", "COGS is {v}."], ["Direct production costs were {v}.", "Goods cost {v} to produce."]),
        ],
        ask_seen: &["What is the gross margin?"],
        ask_unseen: &["What share of sales is left after direct costs?"],
    },
    OpDef {
        op: "nmargin",
        dom: 1,
        sub: 2,
        slots: &[
            sl!("ni", K::Money, ["Net income is {v}.", "The company earned a net income of {v}."], ["Profit after everything came to {v}.", "Bottom-line earnings were {v}."]),
            sl!("rev", K::Money, ["Revenue is {v}.", "Sales were {v}."], ["Turnover came to {v}.", "Total takings were {v}."]),
        ],
        ask_seen: &["What is the net profit margin?"],
        ask_unseen: &["What fraction of sales ends up as profit?"],
    },
    OpDef {
        op: "pctchg",
        dom: 1,
        sub: 3,
        slots: &[
            sl!("old", K::Money, ["The old value is {v}.", "Last year the figure was {v}."], ["Previously it stood at {v}.", "Before the change it read {v}."]),
            sl!("new", K::Money, ["The new value is {v}.", "This year the figure is {v}."], ["Now it reads {v}.", "It has since moved to {v}."]),
        ],
        ask_seen: &["What is the percentage change?"],
        ask_unseen: &["By what percent did it move?"],
    },
    OpDef {
        op: "grow",
        dom: 1,
        sub: 3,
        slots: &[
            sl!("base", K::Money, ["The starting amount is {v}.", "A quantity starts at {v}."], ["It begins at {v}.", "The initial level is {v}."]),
            sl!("rate", K::Pct, ["It grows {v} per year.", "The annual growth rate is {v}."], ["Each year it rises by {v}.", "Yearly expansion is {v}."]),
            sl!("n", K::Count, ["This continues for {v} years.", "The period is {v} years."], ["Over {v} years in a row.", "Keep going for {v} years."]),
        ],
        ask_seen: &["What is the value after growth?"],
        ask_unseen: &["Where does it end up?"],
    },
    OpDef {
        op: "elast",
        dom: 1,
        sub: 3,
        slots: &[
            sl!("q1", K::Money, ["The initial quantity demanded is {v}."], ["At first people buy {v} units."]),
            sl!("q2", K::Money, ["The new quantity demanded is {v}."], ["Afterwards they buy {v} units."]),
            sl!("p1", K::Money, ["The initial price is {v}."], ["The price used to be {v}."]),
            sl!("p2", K::Money, ["The new price is {v}."], ["The price becomes {v}."]),
        ],
        ask_seen: &["What is the price elasticity of demand (midpoint method)?"],
        ask_unseen: &["How responsive is demand to the price move?"],
    },
    OpDef {
        op: "real",
        dom: 1,
        sub: 3,
        slots: &[
            sl!("nominal", K::Money, ["The nominal amount is {v}.", "The nominal wage is {v}."], ["The pay packet reads {v}.", "The stated figure is {v}."]),
            sl!("cpi", K::Num, ["The current CPI is {v}."], ["The price index now reads {v}."]),
            sl!("base", K::Num, ["The base year CPI is {v}."], ["The reference index is {v}."]),
        ],
        ask_seen: &["What is the real value in base-year terms?"],
        ask_unseen: &["What is it worth after adjusting for price changes?"],
    },
    OpDef {
        op: "infl",
        dom: 1,
        sub: 3,
        slots: &[sl!("cpi0", K::Num, ["The CPI last year was {v}."], ["The price index a year ago stood at {v}."]), sl!("cpi1", K::Num, ["The CPI this year is {v}."], ["The index now reads {v}."])],
        ask_seen: &["What is the inflation rate?"],
        ask_unseen: &["How fast did prices rise?"],
    },
    OpDef {
        op: "uconv",
        dom: 2,
        sub: 4,
        slots: &[
            sl!("x", K::Num, ["Convert {v} {from} to {to}.", "How many {to} are in {v} {from}?"], ["Express {v} {from} in {to}.", "A reading of {v} {from} needs to be given in {to}."]),
            sli!("from", K::Unit, []),
            sli!("to", K::Unit, []),
        ],
        ask_seen: &[],
        ask_unseen: &[],
    },
    OpDef {
        op: "sfround",
        dom: 2,
        sub: 5,
        slots: &[sl!("x", K::Num, ["Round {v} to {n} significant figures."], ["Give {v} correct to {n} significant digits."]), sli!("n", K::Count, ["significant"])],
        ask_seen: &[],
        ask_unseen: &[],
    },
    OpDef { op: "sci", dom: 2, sub: 5, slots: &[sl!("x", K::Num, ["Write {v} in scientific notation."], ["Express {v} in standard form."])], ask_seen: &[], ask_unseen: &[] },
    OpDef {
        op: "errprop",
        dom: 2,
        sub: 5,
        slots: &[
            Slot { key: "f", kind: K::E, opt: false, seen: &["What is the uncertainty in the {v} of the two measurements?"], unseen: &["How uncertain is the {v} of these two readings?"], en: EN_ERR, cue: &[] },
            sl!("x", K::Num, ["The first measurement is {v} +/- {sx}."], ["One reading gives {v} plus or minus {sx}."]),
            sli!("sx", K::Num, ["+/-"]),
            sl!("y", K::Num, ["The second measurement is {v} +/- {sy}."], ["Another reading gives {v} plus or minus {sy}."]),
            sli!("sy", K::Num, ["+/-"]),
        ],
        ask_seen: &[],
        ask_unseen: &[],
    },
    OpDef {
        op: "mean",
        dom: 2,
        sub: 5,
        slots: &[sl!("x", K::List, ["The measurements are {v}."], ["The readings come out as {v}."])],
        ask_seen: &["What is the mean?", "Find the average."],
        ask_unseen: &["What is the typical value across these?"],
    },
    OpDef {
        op: "sd",
        dom: 2,
        sub: 5,
        slots: &[sl!("x", K::List, ["The measurements are {v}."], ["The readings come out as {v}."])],
        ask_seen: &["What is the sample standard deviation?"],
        ask_unseen: &["How spread out are the readings (sample)?"],
    },
    OpDef {
        op: "ci95",
        dom: 2,
        sub: 5,
        slots: &[sl!("x", K::List, ["The measurements are {v}."], ["The readings come out as {v}."])],
        ask_seen: &["Give the 95% confidence interval half-width for the mean."],
        ask_unseen: &["How wide is the margin of error around the average at 95% confidence?"],
    },
    OpDef {
        op: "slope",
        dom: 2,
        sub: 5,
        slots: &[sl!("x", K::List, ["The x values are {v}."], ["Inputs: {v}."]), sl!("y", K::List, ["The y values are {v}."], ["Outputs: {v}."])],
        ask_seen: &["What is the slope of the least-squares line?"],
        ask_unseen: &["By how much does y change per unit of x in the best-fit line?"],
    },
    OpDef {
        op: "molmass",
        dom: 2,
        sub: 4,
        slots: &[sl!("f", K::Formula, ["What is the molar mass of {v}?", "Find the molar mass of {v} in g/mol."], ["How heavy is one mole of {v}?", "What does one mole of {v} weigh in grams?"])],
        ask_seen: &[],
        ask_unseen: &[],
    },
    OpDef {
        op: "stoich",
        dom: 2,
        sub: 4,
        slots: &[
            sl!("eq", K::Eq, ["Consider the reaction {v}."], ["The process is {v}."]),
            sl!("m", K::Num, ["How many grams of {want} can be made from {v} g of {given}?"], ["Starting from {v} g of {given}, what mass of {want} is obtained?"]),
            sli!("given", K::Species, []),
            sli!("want", K::Species, []),
        ],
        ask_seen: &[],
        ask_unseen: &[],
    },
    OpDef {
        op: "gas",
        dom: 2,
        sub: 4,
        slots: &[
            Slot { key: "find", kind: K::E, opt: false, seen: &["Find the {v}."], unseen: &["Work out the {v}."], en: EN_GAS, cue: &[] },
            slo!("p", K::Num, ["The pressure is {v} Pa."], ["It sits at {v} pascals."]),
            slo!("v", K::Num, ["The volume is {v} m^3."], ["The container holds {v} cubic metres."]),
            slo!("n", K::Num, ["There are {v} mol of gas."], ["The sample contains {v} moles."]),
            slo!("t", K::Num, ["The temperature is {v} K."], ["It is held at {v} kelvin."]),
        ],
        ask_seen: &[],
        ask_unseen: &[],
    },
    OpDef {
        op: "kin",
        dom: 2,
        sub: 4,
        slots: &[
            Slot { key: "find", kind: K::E, opt: false, seen: &["Find the {v}."], unseen: &["Work out the {v}."], en: EN_KIN, cue: &[] },
            slo!("u", K::Num, ["The initial velocity is {v} m/s."], ["It starts at {v} metres per second."]),
            slo!("v", K::Num, ["The final velocity is {v} m/s."], ["It ends at {v} metres per second."]),
            slo!("a", K::Num, ["The acceleration is {v} m/s^2."], ["It speeds up at {v} metres per second squared."]),
            slo!("t", K::Num, ["The time taken is {v} s."], ["This lasts {v} seconds."]),
            slo!("s", K::Num, ["The distance covered is {v} m."], ["The displacement is {v} metres."]),
        ],
        ask_seen: &[],
        ask_unseen: &[],
    },
];

pub fn op_def(op: &str) -> Option<&'static OpDef> {
    OPS.iter().find(|o| o.op == op)
}

// ---------------------------------------------------------------------------------------------
// units

pub struct UnitW {
    pub spec: &'static str,
    pub seen: &'static [&'static str],
    pub unseen: &'static [&'static str],
}
const fn uw(spec: &'static str, seen: &'static [&'static str], unseen: &'static [&'static str]) -> UnitW {
    UnitW { spec, seen, unseen }
}
pub static UNITS: &[UnitW] = &[
    uw("m", &["m", "metres"], &["meters"]),
    uw("km", &["km", "kilometres"], &["kilometers", "klicks"]),
    uw("cm", &["cm", "centimetres"], &["centimeters"]),
    uw("mm", &["mm", "millimetres"], &["millimeters"]),
    uw("in", &["in", "inches"], &["inch"]),
    uw("ft", &["ft", "feet"], &["foot"]),
    uw("yd", &["yd", "yards"], &["yard"]),
    uw("mi", &["mi", "miles"], &["mile"]),
    uw("kg", &["kg", "kilograms"], &["kilos"]),
    uw("g", &["g", "grams"], &["gram"]),
    uw("mg", &["mg", "milligrams"], &["milligram"]),
    uw("lb", &["lb", "pounds"], &["lbs"]),
    uw("oz", &["oz", "ounces"], &["ounce"]),
    uw("s", &["s", "seconds"], &["secs"]),
    uw("min", &["min", "minutes"], &["mins"]),
    uw("h", &["h", "hours"], &["hrs"]),
    uw("day", &["day", "days"], &["d"]),
    uw("m/s", &["m/s", "metres per second"], &["meters per second", "mps"]),
    uw("km/h", &["km/h", "kilometres per hour"], &["kph", "kilometers an hour"]),
    uw("mi/h", &["mi/h", "miles per hour"], &["mph"]),
    uw("ft/s", &["ft/s", "feet per second"], &["fps"]),
    uw("knot", &["knot", "knots"], &["kn"]),
    uw("Pa", &["Pa", "pascals"], &["pascal"]),
    uw("kPa", &["kPa", "kilopascals"], &["kilopascal"]),
    uw("bar", &["bar", "bars"], &["bar"]),
    uw("atm", &["atm", "atmospheres"], &["atmosphere"]),
    uw("psi", &["psi", "pounds per square inch"], &["lbf/in^2"]),
    uw("J", &["J", "joules"], &["joule"]),
    uw("kJ", &["kJ", "kilojoules"], &["kilojoule"]),
    uw("cal", &["cal", "calories"], &["calorie"]),
    uw("kcal", &["kcal", "kilocalories"], &["kilocalorie"]),
    uw("kWh", &["kWh", "kilowatt-hours"], &["kilowatt hours"]),
    uw("W", &["W", "watts"], &["watt"]),
    uw("kW", &["kW", "kilowatts"], &["kilowatt"]),
    uw("N", &["N", "newtons"], &["newton"]),
    uw("kN", &["kN", "kilonewtons"], &["kilonewton"]),
    uw("L", &["L", "litres"], &["liters"]),
    uw("mL", &["mL", "millilitres"], &["milliliters"]),
    uw("m^3", &["m^3", "cubic metres"], &["cubic meters"]),
    uw("kg*m/s^2", &["kg*m/s^2", "kilogram-metres per second squared"], &["kg m s^-2"]),
    uw("N*m", &["N*m", "newton-metres"], &["newton meters"]),
    uw("J/s", &["J/s", "joules per second"], &["joules a second"]),
    uw("kg/m^3", &["kg/m^3", "kilograms per cubic metre"], &["kg per m^3"]),
    uw("g/cm^3", &["g/cm^3", "grams per cubic centimetre"], &["g per cm^3"]),
    uw("m/s^2", &["m/s^2", "metres per second squared"], &["m per s^2"]),
];
/// conversion groups, easy -> hard
pub const UGROUPS: &[&[&str]] = &[
    &["m", "km", "cm", "mm", "in", "ft", "yd", "mi"],
    &["kg", "g", "mg", "lb", "oz"],
    &["s", "min", "h", "day"],
    &["m/s", "km/h", "mi/h", "ft/s", "knot"],
    &["Pa", "kPa", "bar", "atm", "psi"],
    &["J", "kJ", "cal", "kcal", "kWh", "N*m"],
    &["W", "kW", "J/s"],
    &["L", "mL", "m^3"],
    &["N", "kN", "kg*m/s^2"],
    &["kg/m^3", "g/cm^3"],
];

// ---------------------------------------------------------------------------------------------
// parameters

#[derive(Clone, Debug)]
pub struct Param {
    pub key: &'static str,
    pub spec: String,
    /// number / token as it appears in text (before kind-specific decoration)
    pub txt: String,
}
#[derive(Clone, Debug)]
pub struct Params {
    pub op: &'static str,
    pub vals: Vec<Param>,
    /// number of arithmetic steps a by-hand solution needs (drives the ungated model error)
    pub steps: u32,
}
impl Params {
    pub fn spec(&self) -> String {
        let mut s = self.op.to_string();
        for p in &self.vals {
            s.push(' ');
            s.push_str(p.key);
            s.push('=');
            s.push_str(&p.spec);
        }
        s
    }
    pub fn get(&self, k: &str) -> Option<&Param> {
        self.vals.iter().find(|p| p.key == k)
    }
}

fn ri(r: &mut Rng, lo: i64, hi: i64) -> i64 {
    lo + r.below((hi - lo + 1) as usize) as i64
}
fn rnd_to(x: f64, step: f64) -> f64 {
    (x / step).round() * step
}
fn pi(k: &'static str, v: i64) -> Param {
    Param { key: k, spec: v.to_string(), txt: v.to_string() }
}
fn pf(k: &'static str, v: f64) -> Param {
    let s = fmt(v);
    Param { key: k, spec: s.clone(), txt: s }
}
/// rate given as a percent number with tenth/quarter resolution: text "8.5", spec 0.085
fn prate(k: &'static str, pct: f64) -> Param {
    Param { key: k, spec: fmt(pct / 100.0), txt: fmt(pct) }
}
fn plist(k: &'static str, xs: &[f64]) -> Param {
    let l: Vec<String> = xs.iter().map(|x| fmt(*x)).collect();
    Param { key: k, spec: l.join(","), txt: l.join(",") }
}
fn pw(k: &'static str, v: &str) -> Param {
    Param { key: k, spec: v.to_string(), txt: v.to_string() }
}

/// dec(x, 1) text with one decimal
fn one_dec(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

pub fn gen_params(op: &'static str, r: &mut Rng, d: f64, style: u8) -> Option<Params> {
    // style 1 (world B): longer lists, longer terms, larger magnitudes
    let big = if style == 1 { 1.5 } else { 1.0 };
    let len_extra = if style == 1 { 1 } else { 0 };
    let mut steps = 1u32;
    let mut v: Vec<Param> = Vec::new();
    match op {
        "npv" | "irr" | "payback" => {
            let n = 3 + (d * 3.0) as usize + r.below(2) + len_extra;
            let c0 = rnd_to(1000.0 + 9000.0 * r.f64() * big, 100.0);
            let mut cf = vec![-c0];
            if op == "irr" {
                let tot = c0 * (1.25 + 1.0 * r.f64());
                for _ in 0..n {
                    cf.push(rnd_to(tot / n as f64 * (0.6 + 0.8 * r.f64()), 10.0).max(10.0));
                }
                if cf[1..].iter().sum::<f64>() <= c0 * 1.05 {
                    *cf.last_mut()? += c0;
                }
            } else if op == "payback" {
                let mut tot = 0.0;
                let mut k = 0;
                while tot < c0 * 1.2 && k < 8 {
                    let c = rnd_to(c0 * (0.15 + 0.35 * r.f64()), 100.0).max(100.0);
                    cf.push(c);
                    tot += c;
                    k += 1;
                }
                if tot < c0 {
                    return None;
                }
            } else {
                for _ in 0..n {
                    cf.push(rnd_to(c0 * (0.2 + 0.5 * r.f64()), 10.0).max(10.0));
                }
            }
            let len = cf.len() as u32;
            if op == "npv" {
                v.push(prate("rate", rnd_to(4.0 + 11.0 * r.f64(), 0.5)));
                v.push(plist("cf", &cf));
                steps = 1 + len;
            } else if op == "irr" {
                v.push(plist("cf", &cf));
                steps = 9 + len;
            } else {
                v.push(plist("cf", &cf));
                steps = 1 + len / 2;
            }
        }
        "compound" => {
            v.push(pi("p", ri(r, 10, (500.0 * big) as i64) * 100));
            v.push(prate("r", rnd_to(2.0 + 7.0 * r.f64(), 0.25)));
            v.push(pi("n", ri(r, 2, 8 + (12.0 * d) as i64)));
            v.push(pi("m", [1, 2, 4, 12][r.below(4)]));
            steps = 3 + (d * 2.0) as u32;
        }
        "annpv" => {
            v.push(pi("pmt", ri(r, 2, (100.0 * big) as i64) * 50));
            v.push(prate("r", rnd_to(3.0 + 6.0 * r.f64(), 0.25)));
            v.push(pi("n", ri(r, 4, 10 + (20.0 * d) as i64)));
            steps = 4 + (d * 3.0) as u32;
        }
        "loanpmt" | "amort" => {
            let n = [12, 24, 36, 48, 60, 120, 180, 240, 360][r.below(9)];
            v.push(pi("p", ri(r, 5, (400.0 * big) as i64) * 1000));
            v.push(prate("r", rnd_to(0.25 + 0.75 * r.f64(), 0.05)));
            v.push(pi("n", n));
            if op == "amort" {
                v.push(pi("k", ri(r, 3, n - 2)));
                steps = 7 + (d * 3.0) as u32;
            } else {
                steps = 4 + (d * 2.0) as u32;
            }
        }
        "bondprice" | "bondyield" | "bondmod" => {
            let f = if r.f64() < 0.5 { 1 } else { 2 };
            let n = ri(r, 2, 6 + (8.0 * d) as i64);
            let face = [100, 500, 1000, 1000, 1000][r.below(5)];
            let cpn = rnd_to(2.0 + 7.0 * r.f64(), 0.5);
            let y = rnd_to(2.0 + 8.0 * r.f64(), 0.5);
            v.push(pi("face", face));
            v.push(prate("cpn", cpn));
            if op == "bondyield" {
                // price from the yield, rounded to cents (truth = tool on the rounded price)
                let spec = format!("bondprice face={face} cpn={} y={} n={n} f={f}", fmt(cpn / 100.0), fmt(y / 100.0));
                let p: f64 = neural_mesh::fin::parse_out(&neural_mesh::tools::Tool::call(&neural_mesh::fin::FinTool::default(), &spec).ok()?)["v"].parse().ok()?;
                v.push(pf("price", (p * 100.0).round() / 100.0));
            } else {
                v.push(prate("y", y));
            }
            v.push(pi("n", n));
            v.push(pi("f", f));
            steps = (3 + (n * f as i64) / 2).min(12) as u32 + if op == "bondprice" { 0 } else { 4 };
        }
        "cagr" => {
            let s = ri(r, 10, (90.0 * big) as i64) * 100;
            let e = rnd_to(s as f64 * (1.2 + 2.3 * r.f64()), 100.0);
            v.push(pi("s", s));
            v.push(pf("e", e));
            v.push(pi("n", ri(r, 2, 6 + (9.0 * d) as i64)));
            steps = 3;
        }
        "wacc" => {
            v.push(pi("e", ri(r, 100, (900.0 * big) as i64)));
            v.push(pi("d", ri(r, 100, (900.0 * big) as i64)));
            v.push(prate("re", rnd_to(8.0 + 8.0 * r.f64(), 0.5)));
            v.push(prate("rd", rnd_to(3.0 + 5.0 * r.f64(), 0.5)));
            v.push(prate("t", rnd_to(15.0 + 20.0 * r.f64(), 1.0)));
            steps = 6;
        }
        "perp" => {
            v.push(pi("c", ri(r, 5, (90.0 * big) as i64) * 10));
            v.push(prate("r", rnd_to(3.0 + 9.0 * r.f64(), 0.5)));
            steps = 1 + (d * 2.0) as u32;
        }
        "payoff" => {
            let k = ri(r, 4, 40) * 5;
            let s = (k as f64 + ri(r, -30, 30) as f64).max(1.0);
            v.push(pw("kind", if r.f64() < 0.5 { "call" } else { "put" }));
            v.push(pw("pos", if r.f64() < 0.5 { "long" } else { "short" }));
            v.push(pf("s", s));
            v.push(pi("k", k));
            v.push(pf("prem", ri(r, 2, 30) as f64 / 2.0));
            steps = 3;
        }
        "breakeven" => {
            let price = ri(r, 10, 200);
            let vc = ((price as f64) * (0.3 + 0.4 * r.f64())).round().max(1.0);
            v.push(pi("fc", ri(r, 10, (400.0 * big) as i64) * 500));
            v.push(pi("price", price));
            v.push(pf("vc", vc));
            steps = 2;
        }
        "current" => {
            v.push(pi("ca", ri(r, 10, 900) * 10));
            v.push(pi("cl", ri(r, 10, 900) * 10));
        }
        "roe" | "nmargin" => {
            let a = ri(r, 10, 900) * 10;
            let b = a + ri(r, 100, 900) * 10;
            if op == "roe" {
                v.push(pi("ni", a));
                v.push(pi("eq", b));
            } else {
                v.push(pi("ni", a));
                v.push(pi("rev", b));
            }
        }
        "gmargin" => {
            let rev = ri(r, 100, 900) * 10;
            let cogs = (rev as f64 * (0.3 + 0.5 * r.f64())).round();
            v.push(pi("rev", rev));
            v.push(pf("cogs", cogs));
            steps = 2;
        }
        "pctchg" => {
            v.push(pi("old", ri(r, 20, 900)));
            v.push(pi("new", ri(r, 20, 900)));
            steps = 2;
        }
        "grow" => {
            v.push(pi("base", ri(r, 10, (900.0 * big) as i64) * 10));
            v.push(Param { key: "rate", spec: fmt(rnd_to(2.0 + 10.0 * r.f64(), 0.5)), txt: String::new() });
            let ratev = v[1].spec.clone();
            v[1].txt = ratev;
            v.push(pi("n", ri(r, 2, 6 + (8.0 * d) as i64)));
            steps = 3 + (d * 3.0) as u32;
        }
        "elast" => {
            let p1 = ri(r, 5, 50);
            let p2 = p1 + ri(r, 1, 10) * if r.f64() < 0.5 { 1 } else { -1 };
            if p2 < 1 || p2 == p1 {
                return None;
            }
            let q1 = ri(r, 100, 900);
            let q2 = (q1 as f64 * (0.6 + 0.8 * r.f64())).round() as i64;
            if q1 + q2 == 0 {
                return None;
            }
            v.push(pi("q1", q1));
            v.push(pi("q2", q2));
            v.push(pi("p1", p1));
            v.push(pi("p2", p2));
            steps = 6;
        }
        "real" => {
            let base = [100.0, 100.0, 120.0, 90.0][r.below(4)];
            v.push(pi("nominal", ri(r, 10, 900) * 10));
            v.push(pf("cpi", one_dec(base * (1.0 + 0.3 * r.f64()))));
            v.push(pf("base", base));
            steps = 2;
        }
        "infl" => {
            let c0 = one_dec(100.0 + 40.0 * r.f64());
            v.push(pf("cpi0", c0));
            v.push(pf("cpi1", one_dec(c0 * (1.0 + 0.1 * r.f64()))));
            steps = 2;
        }
        "uconv" => {
            let gmax = (2.0 + d * 8.0) as usize;
            let g = UGROUPS[r.below(gmax.min(UGROUPS.len()))];
            let a = r.below(g.len());
            let mut b = r.below(g.len());
            if a == b {
                b = (b + 1) % g.len();
            }
            let x = match r.below(3) {
                0 => format!("{}", ri(r, 2, 900)),
                1 => format!("{}.{}", ri(r, 1, 99), ri(r, 1, 9)),
                _ => format!("{}", ri(r, 10, 5000)),
            };
            v.push(Param { key: "x", spec: x.clone(), txt: x });
            v.push(pw("from", g[a]));
            v.push(pw("to", g[b]));
            steps = 2 + (d * 3.0) as u32 + if g[a].contains('/') || g[a].contains('*') { 2 } else { 0 };
        }
        "sfround" | "sci" => {
            let digs = ri(r, 4, 7) as usize;
            let mut ds: Vec<u8> = (0..digs).map(|_| r.below(10) as u8).collect();
            ds[0] = 1 + r.below(9) as u8;
            let intlen = ri(r, 0, 3) as usize;
            let s: String = ds.iter().map(|d| (b'0' + d) as char).collect();
            let x = if r.f64() < 0.5 {
                format!("0.{}{}", "0".repeat(ri(r, 1, 3) as usize), s)
            } else if intlen > 0 && intlen < digs {
                format!("{}.{}", &s[..intlen], &s[intlen..])
            } else {
                s.clone()
            };
            if op == "sfround" {
                let n = ri(r, 2, 3);
                // avoid exact-half ties at the rounding digit
                let sig: Vec<u8> = x.bytes().filter(|c| c.is_ascii_digit()).skip_while(|c| *c == b'0').collect();
                if sig.len() > n as usize && sig[n as usize] == b'5' && sig[n as usize + 1..].iter().all(|c| *c == b'0') {
                    return None;
                }
                v.push(Param { key: "x", spec: x.clone(), txt: x });
                v.push(pi("n", n));
                steps = 2;
            } else {
                v.push(Param { key: "x", spec: x.clone(), txt: x });
                steps = 2;
            }
        }
        "errprop" => {
            let x = ri(r, 10, 90) as f64;
            let y = ri(r, 5, 60) as f64;
            v.push(pw("f", if r.f64() < 0.5 { "add" } else { "mul" }));
            v.push(pf("x", x));
            v.push(pf("sx", ri(r, 1, 20) as f64 / 10.0));
            v.push(pf("y", y));
            v.push(pf("sy", ri(r, 1, 20) as f64 / 10.0));
            steps = if v[0].spec == "add" { 4 } else { 7 };
        }
        "mean" | "sd" | "ci95" => {
            let n = ri(r, 5, 8 + (4.0 * d) as i64 + len_extra as i64) as usize;
            let xs: Vec<f64> = (0..n).map(|_| one_dec(10.0 + 90.0 * r.f64())).collect();
            v.push(plist("x", &xs));
            steps = match op {
                "mean" => 1 + n as u32 / 3,
                "sd" => 3 + n as u32 / 2,
                _ => 5 + n as u32 / 2,
            };
        }
        "slope" => {
            let n = ri(r, 5, 7 + len_extra as i64) as usize;
            let a = ri(r, -5, 9) as f64 / 2.0;
            let b = ri(r, 0, 20) as f64;
            let xs: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            let ys: Vec<f64> = xs.iter().map(|x| one_dec(a * x + b + (r.f64() - 0.5) * 4.0)).collect();
            v.push(plist("x", &xs));
            v.push(plist("y", &ys));
            steps = 4 + n as u32;
        }
        "molmass" => {
            let fs = ["H2O", "CO2", "NaCl", "NH3", "CH4", "H2SO4", "CaCO3", "C6H12O6", "Ca(OH)2", "Al2(SO4)3", "Mg(NO3)2", "Fe2O3", "C2H5OH", "NaHCO3", "CuSO4", "KMnO4", "Ca3(PO4)2", "C12H22O11"];
            let hi = (6.0 + 12.0 * d) as usize;
            let f = fs[r.below(hi.min(fs.len()))];
            v.push(pw("f", f));
            steps = 2 + f.chars().filter(|c| c.is_ascii_uppercase()).count() as u32;
        }
        "stoich" => {
            let rx = [
                ("2H2+O2>2H2O", "2 H2 + O2 -> 2 H2O"),
                ("N2+3H2>2NH3", "N2 + 3 H2 -> 2 NH3"),
                ("CH4+2O2>CO2+2H2O", "CH4 + 2 O2 -> CO2 + 2 H2O"),
                ("2Mg+O2>2MgO", "2 Mg + O2 -> 2 MgO"),
                ("C3H8+5O2>3CO2+4H2O", "C3H8 + 5 O2 -> 3 CO2 + 4 H2O"),
                ("2Al+3Cl2>2AlCl3", "2 Al + 3 Cl2 -> 2 AlCl3"),
                ("CaCO3>CaO+CO2", "CaCO3 -> CaO + CO2"),
            ];
            let (eq, _txt) = rx[r.below(rx.len())];
            let (l, rr) = eq.split_once('>')?;
            let strip = |t: &str| -> String { t.trim_start_matches(|c: char| c.is_ascii_digit()).to_string() };
            let lhs: Vec<String> = l.split('+').map(strip).collect();
            let rhs: Vec<String> = rr.split('+').map(strip).collect();
            let given = lhs[r.below(lhs.len())].clone();
            let want = rhs[r.below(rhs.len())].clone();
            v.push(Param { key: "eq", spec: eq.to_string(), txt: _txt.to_string() });
            v.push(pw("given", &given));
            v.push(pf("m", ri(r, 20, 1000) as f64 / 10.0));
            v.push(pw("want", &want));
            steps = 5 + (d * 2.0) as u32;
        }
        "gas" => {
            let find = ["p", "v", "n", "t"][r.below(4)];
            let p = ri(r, 50, 300) * 1000;
            let vv = ri(r, 5, 100) as f64 / 1000.0;
            let n = ri(r, 5, 50) as f64 / 10.0;
            let t = ri(r, 250, 400);
            v.push(pw("find", find));
            if find != "p" {
                v.push(pi("p", p));
            }
            if find != "v" {
                v.push(pf("v", vv));
            }
            if find != "n" {
                v.push(pf("n", n));
            }
            if find != "t" {
                v.push(pi("t", t));
            }
            steps = 3 + (d * 2.0) as u32;
        }
        "kin" => {
            // choose a consistent (u, a, t) scenario then expose three of the five
            let u = ri(r, 0, 20) as f64;
            let a = ri(r, 1, 12) as f64 / 2.0;
            let t = ri(r, 2, 15) as f64;
            let vf = u + a * t;
            let s = u * t + 0.5 * a * t * t;
            let sets: [(&[&str], &str); 5] = [(&["u", "a", "t"], "s"), (&["u", "a", "t"], "v"), (&["u", "v", "t"], "a"), (&["u", "v", "a"], "t"), (&["u", "a", "s"], "v")];
            let (given, find) = sets[r.below(sets.len())];
            v.push(pw("find", find));
            for g in given {
                let val = match *g {
                    "u" => u,
                    "v" => vf,
                    "a" => a,
                    "t" => t,
                    _ => s,
                };
                let key: &'static str = match *g {
                    "u" => "u",
                    "v" => "v",
                    "a" => "a",
                    "t" => "t",
                    _ => "s",
                };
                v.push(pf(key, val));
            }
            steps = 4 + (d * 3.0) as u32;
        }
        _ => return None,
    }
    // canonical key order for stable specs: keep as generated
    Some(Params { op, vals: v, steps })
}

// ---------------------------------------------------------------------------------------------
// rendering

pub const NUMWORDS: [&str; 21] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
];

fn thousands(s: &str) -> String {
    // s like "1234567.89" or "-1200"
    let neg = s.starts_with('-');
    let t = s.trim_start_matches('-');
    let (ip, fp) = match t.split_once('.') {
        Some((a, b)) => (a, Some(b)),
        None => (t, None),
    };
    let mut o = String::new();
    for (i, c) in ip.chars().enumerate() {
        if i > 0 && (ip.len() - i) % 3 == 0 {
            o.push(',');
        }
        o.push(c);
    }
    let mut s2 = String::new();
    if neg {
        s2.push('-');
    }
    s2.push_str(&o);
    if let Some(f) = fp {
        s2.push('.');
        s2.push_str(f);
    }
    s2
}

pub struct RCfg {
    pub level: u8,
    pub unseen: bool,
}

fn unit_surface(spec: &str, unseen: bool, r: &mut Rng) -> String {
    if let Some(u) = UNITS.iter().find(|u| u.spec == spec) {
        if unseen && r.f64() < 0.8 && !u.unseen.is_empty() {
            return u.unseen[r.below(u.unseen.len())].to_string();
        }
        return u.seen[r.below(u.seen.len())].to_string();
    }
    spec.to_string()
}

fn render_val(slot: &Slot, p: &Param, params: &Params, cfg: &RCfg, r: &mut Rng) -> String {
    let lvl = cfg.level;
    match slot.kind {
        K::Rate | K::Pct => {
            let t = &p.txt;
            if lvl >= 1 && r.f64() < 0.5 {
                if cfg.unseen { format!("{t} per cent") } else { format!("{t} percent") }
            } else {
                format!("{t}%")
            }
        }
        K::Money => {
            let t = &p.txt;
            if lvl >= 1 && r.f64() < 0.7 {
                let th = thousands(t);
                if r.f64() < 0.6 { format!("${th}") } else if cfg.unseen { format!("{th} dollars") } else { th }
            } else {
                t.clone()
            }
        }
        K::Count => {
            if lvl >= 2 {
                if let Ok(n) = p.txt.parse::<usize>() {
                    if n <= 20 && r.f64() < 0.6 {
                        return NUMWORDS[n].to_string();
                    }
                }
            }
            p.txt.clone()
        }
        K::Num => p.txt.clone(),
        K::List => {
            let xs: Vec<&str> = p.txt.split(',').collect();
            let neg_first = xs.first().map(|s| s.starts_with('-')).unwrap_or(false);
            let joiner = |items: &[String]| -> String {
                if items.len() == 1 {
                    items[0].clone()
                } else {
                    format!("{} and {}", items[..items.len() - 1].join(", "), items[items.len() - 1])
                }
            };
            if neg_first && (params.op == "npv" || params.op == "irr" || params.op == "payback") && lvl >= 1 && r.f64() < 0.5 {
                // outlay + inflows form
                let m = |s: &str| -> String { format!("${}", thousands(s.trim_start_matches('-'))) };
                let out = m(xs[0]);
                let inf: Vec<String> = xs[1..].iter().map(|s| m(s)).collect();
                return if cfg.unseen {
                    format!("an outlay of {out} followed by receipts of {}", joiner(&inf))
                } else {
                    format!("an initial investment of {out} followed by inflows of {}", joiner(&inf))
                };
            }
            if lvl >= 1 && r.f64() < 0.3 {
                xs.join(", ")
            } else {
                xs.join(", ")
            }
        }
        K::Unit => unit_surface(&p.spec, cfg.unseen, r),
        K::Formula | K::Species => p.txt.clone(),
        K::Eq => p.txt.clone(),
        K::E => {
            let e = slot.en.iter().find(|e| e.val == p.spec);
            match e {
                Some(e) => {
                    if cfg.unseen && r.f64() < 0.85 {
                        e.unseen[r.below(e.unseen.len())].to_string()
                    } else {
                        e.seen[r.below(e.seen.len())].to_string()
                    }
                }
                None => p.spec.clone(),
            }
        }
    }
}

fn typo(w: &str, r: &mut Rng) -> String {
    let c: Vec<char> = w.chars().collect();
    if c.len() < 5 || !c.iter().all(|x| x.is_ascii_alphabetic()) {
        return w.to_string();
    }
    let i = 1 + r.below(c.len() - 3);
    let mut o = c.clone();
    if r.f64() < 0.6 {
        o.swap(i, i + 1);
    } else {
        o.remove(i);
    }
    o.into_iter().collect()
}

pub const DIST_SEEN: &[&str] = &["On March 14, 2024 the request was logged.", "The reference is ticket #4471.", "Answer within 30 seconds.", "See page 12 of the handbook."];
pub const DIST_UNSEEN: &[&str] = &["Order no. 8812 refers to this.", "Q3 2023 was the last review.", "It should take about 15 minutes of work."];
pub const FILLER_SEEN: &[&str] = &["Please show the final answer.", "Give the answer clearly."];
pub const FILLER_UNSEEN: &[&str] = &["A short reply is fine.", "Thanks in advance for the figure."];

/// the text of a word problem for `params`; also reports what was dropped at level 4
pub fn render(def: &OpDef, params: &Params, cfg: &RCfg, r: &mut Rng) -> (String, bool) {
    let mut sents: Vec<String> = Vec::new();
    let mut asks: Vec<String> = Vec::new();
    let pool = |seen: &'static [&'static str], unseen: &'static [&'static str], r: &mut Rng| -> Option<&'static str> {
        if cfg.unseen && !unseen.is_empty() && r.f64() < 0.85 {
            Some(unseen[r.below(unseen.len())])
        } else if !seen.is_empty() {
            Some(seen[r.below(seen.len())])
        } else if !unseen.is_empty() {
            Some(unseen[r.below(unseen.len())])
        } else {
            None
        }
    };
    // pre-render every value once so inline references are consistent
    let mut rendered: Vec<(&str, String)> = Vec::new();
    for s in def.slots {
        if let Some(p) = params.get(s.key) {
            rendered.push((s.key, render_val(s, p, params, cfg, r)));
        }
    }
    let rv = |k: &str| -> String { rendered.iter().find(|(kk, _)| *kk == k).map(|(_, v)| v.clone()).unwrap_or_default() };
    let mut dropped = false;
    let numeric_slots: Vec<usize> = def.slots.iter().enumerate().filter(|(_, s)| !s.seen.is_empty() && !matches!(s.kind, K::E) && params.get(s.key).is_some()).map(|(i, _)| i).collect();
    let mut drop_idx: Option<usize> = None;
    if cfg.level == 4 && numeric_slots.len() >= 2 && r.f64() < 0.6 {
        drop_idx = Some(numeric_slots[r.below(numeric_slots.len())]);
        dropped = true;
    }
    let mut find_sentence: Option<String> = None;
    for (i, s) in def.slots.iter().enumerate() {
        let p = match params.get(s.key) {
            Some(p) => p,
            None => continue,
        };
        if drop_idx == Some(i) {
            continue;
        }
        let ph = match pool(s.seen, s.unseen, r) {
            Some(x) => x,
            None => continue,
        };
        let mut t = ph.replace("{v}", &rv(s.key));
        for (k, v) in &rendered {
            t = t.replace(&format!("{{{k}}}"), v);
        }
        let _ = p;
        // outlay form replaces the whole sentence when rendered that way
        if s.kind == K::List && rv(s.key).starts_with("an ") {
            t = format!("The project has {}.", rv(s.key));
        }
        if s.key == "find" {
            find_sentence = Some(t);
        } else {
            sents.push(t);
        }
    }
    // e-slot sentences that are questions (errprop f) go last
    if def.op == "errprop" {
        if let Some(pos) = sents.iter().position(|s| s.starts_with("What is the uncertainty") || s.starts_with("How uncertain")) {
            let q = sents.remove(pos);
            find_sentence = Some(q);
        }
    }
    let ask = pool(def.ask_seen, def.ask_unseen, r);
    if let Some(a) = ask {
        asks.push(a.to_string());
    }
    if cfg.level >= 1 {
        // shuffle statement order (Fisher-Yates)
        for i in (1..sents.len()).rev() {
            let j = r.below(i + 1);
            sents.swap(i, j);
        }
    }
    if cfg.level == 4 && !dropped && !numeric_slots.is_empty() {
        // conflicting second value for one of the numeric slots
        let s = &def.slots[numeric_slots[r.below(numeric_slots.len())]];
        if matches!(s.kind, K::Rate | K::Pct) {
            if let Some(ph) = pool(s.seen, s.unseen, r) {
                let other = format!("{:.1}%", 2.0 + 10.0 * r.f64());
                sents.push(format!("Earlier reports said something different: {}", ph.replace("{v}", &other).to_lowercase()));
            }
        } else if matches!(s.kind, K::Money | K::Count) {
            if let Some(ph) = pool(s.seen, s.unseen, r) {
                let other = format!("{}", 100 + r.below(900) * 10);
                let t = ph.replace("{v}", &other);
                let mut t2 = t.clone();
                for (k, _) in &rendered {
                    t2 = t2.replace(&format!("{{{k}}}"), "");
                }
                sents.push(format!("An older note claimed this: {}", t2));
            }
        }
    }
    let lead = if r.f64() < 0.6 {
        let pool_l = if cfg.unseen { LEADS_UNSEEN[def.dom] } else { LEADS_SEEN[def.dom] };
        Some(pool_l[r.below(pool_l.len())].to_string())
    } else {
        None
    };
    let mut out: Vec<String> = Vec::new();
    if let Some(l) = lead {
        out.push(l);
    }
    out.extend(sents);
    if cfg.level >= 2 {
        let d = if cfg.unseen && r.f64() < 0.7 { DIST_UNSEEN[r.below(DIST_UNSEEN.len())] } else { DIST_SEEN[r.below(DIST_SEEN.len())] };
        let pos = r.below(out.len() + 1);
        out.insert(pos, d.to_string());
    }
    if let Some(f) = find_sentence {
        out.push(f);
    }
    out.extend(asks);
    if cfg.level >= 1 && r.f64() < 0.4 {
        let f = if cfg.unseen { FILLER_UNSEEN[r.below(FILLER_UNSEEN.len())] } else { FILLER_SEEN[r.below(FILLER_SEEN.len())] };
        out.push(f.to_string());
    }
    let mut text = out.join(" ");
    if cfg.level == 3 {
        // typos on prose words only (skip anything with digits / symbols / capitals)
        let ws: Vec<String> = text.split(' ').map(|w| w.to_string()).collect();
        let mut o = Vec::new();
        for w in ws {
            let core: String = w.chars().filter(|c| c.is_ascii_alphabetic()).collect();
            let tail: String = w.chars().skip_while(|c| c.is_ascii_alphabetic()).collect();
            if core.len() == w.len() - tail.len() && core.chars().all(|c| c.is_ascii_lowercase()) && r.f64() < 0.12 {
                o.push(format!("{}{}", typo(&core, r), tail));
            } else {
                o.push(w);
            }
        }
        text = o.join(" ");
    }
    (text, dropped)
}

/// prose / hard-negative tasks: no exact computation needed (some contain arithmetic-looking fragments)
pub fn gen_prose(r: &mut Rng, unseen: bool) -> String {
    let topics = ["the trade-offs of leasing versus buying equipment", "how a central bank signals policy", "why diversification reduces risk", "the history of double-entry bookkeeping", "what makes a good lab notebook entry", "the role of peer review in science", "how to explain compound interest to a teenager", "the main assumptions behind the NPV method", "why units must be checked in physics", "the difference between accuracy and precision"];
    let t = topics[r.below(topics.len())];
    let seen = [
        format!("Explain {t}."),
        format!("Discuss {t} in two paragraphs."),
        format!("Summarise {t} for a new colleague."),
        format!("Explain why 12 plus 7 is used as the classic first example when teaching {t}."),
        format!("In 2019 the team of 3 analysts wrote about {t}. Summarise the main idea."),
        format!("Which of the following best describes {t}? Option A, B, C or D."),
    ];
    let un = [
        format!("Could you give me a feel for {t}?"),
        format!("Write a short note on {t}, no calculations needed."),
        format!("Why do people argue about {t}? About 5 sentences please."),
        format!("A friend asked about {t} and 2 related ideas. What would you say?"),
    ];
    if unseen {
        un[r.below(un.len())].clone()
    } else {
        seen[r.below(seen.len())].clone()
    }
}
